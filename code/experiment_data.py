"""Data and response-mask contracts shared by training and evaluation."""
import hashlib
import json
import re
import zipfile
from collections import Counter

BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
TEMPLATE_HASH = 'cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f'
HASHES = {
    'train.jsonl': 'f2851c4d747445d727e86fd03483bfce30250b0843db25b38f8a86d3a4aeaab3',
    'heldout.jsonl': '87fb5abbb936b05909856c6f22ec56fdc84d106b0d737aee3e0d49b60bcf1002',
}
COUNTS = {'train.jsonl': 1400, 'heldout.jsonl': 100}


def jsonl(raw):
    lines = raw.decode('utf-8').split('\n')
    while lines and not lines[-1]:
        lines.pop()
    return [json.loads(line) for line in lines]


def row_key(row):
    return json.dumps({k: row[k] for k in ('instruction', 'input', 'output')}, sort_keys=True, ensure_ascii=False)


def dedupe_report(train, heldout):
    a, b = [Counter(row_key(r) for r in rows) for rows in (train, heldout)]
    tokenize = lambda s: set(re.findall(r'\w+', s.casefold()))
    train_tokens = [tokenize(r['instruction']) for r in train]
    pairs = []
    for j, row in enumerate(heldout):
        tokens = tokenize(row['instruction'])
        for i, other in enumerate(train_tokens):
            union = tokens | other
            similarity = len(tokens & other) / len(union) if union else 1.0
            if similarity >= 0.8:
                pairs.append({'train_index': i, 'heldout_index': j, 'jaccard': similarity})
    return {'exact_overlap': len(a.keys() & b.keys()),
            'duplicate_rows': sum(v-1 for c in (a,b) for v in c.values()),
            'duplicate_train_instructions': sum(v-1 for v in Counter(r['instruction'] for r in train).values()),
            'near_instruction_pairs': pairs}


def read_data(path, hashes=None, counts=None):
    hashes = HASHES if hashes is None else hashes
    counts = COUNTS if counts is None else counts
    result = []
    with zipfile.ZipFile(path) as archive:
        for name, count in counts.items():
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != hashes[name]:
                raise ValueError('Dataset hash mismatch: ' + name)
            rows = jsonl(raw)
            if len(rows) != count or any(
                not all(isinstance(r.get(k), str) for k in ['instruction', 'input', 'output'])
                or not r['instruction'].strip() or not r['output'].strip() for r in rows
            ):
                raise ValueError('Invalid dataset schema or count: ' + name)
            result.append(rows)
    keys = [row_key(r) for rows in result for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate or shared train/heldout row')
    return result


def encode(tok, row, max_len=512, supervise_trailing_newline=False):
    user = row['instruction'] + ('\n\n' + row['input'] if row['input'] else '')
    messages = [{'role': 'user', 'content': user}]
    prompt = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tok.apply_chat_template(messages + [{'role': 'assistant', 'content': row['output']}], tokenize=True, return_dict=False)
    if not all(isinstance(ids, list) and all(type(i) is int for i in ids) for ids in (prompt, full)):
        raise ValueError('Tokenizer did not return list[int] with return_dict=False')
    if full[:len(prompt)] != prompt or not len(prompt) < len(full) <= max_len:
        raise ValueError(f'Template prefix, target length or {max_len}-token limit failed')
    eos_positions = [i for i in range(len(prompt), len(full)) if full[i] == tok.eos_token_id]
    if not eos_positions:
        raise ValueError('Assistant EOS missing')
    labels = [-100] * len(prompt) + full[len(prompt):]
    if not supervise_trailing_newline:
        last = eos_positions[-1]
        labels[last+1:] = [-100] * (len(full)-last-1)
    return {'input_ids': full, 'attention_mask': [1] * len(full), 'labels': labels}


def mask_audit(tok, rows, max_len=512, supervise_trailing_newline=False):
    cases = []
    for i, row in enumerate(rows):
        try:
            item = encode(tok, row, max_len, supervise_trailing_newline)
            labels = [t for t in item['labels'] if t != -100]
            text = tok.decode(labels, skip_special_tokens=False)
            cases.append({'index': i, 'target_tokens': len(labels), 'role_token_in_labels': '<|im_start|>' in text,
                          'eos_last_supervised': bool(labels) and labels[-1] == tok.eos_token_id, 'error': None})
        except ValueError as error:
            cases.append({'index': i, 'target_tokens': 0, 'role_token_in_labels': None,
                          'eos_last_supervised': False, 'error': str(error)})
    return {'rows': cases, 'zero_target_rows': sum(c['target_tokens'] == 0 for c in cases),
            'role_token_rows': sum(c['role_token_in_labels'] is True for c in cases),
            'eos_not_last_rows': sum(not c['eos_last_supervised'] for c in cases)}
