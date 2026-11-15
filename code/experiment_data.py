"""Shared data validation and response-only tokenization for the NEW rerun."""
import hashlib
import json
import zipfile

BASE = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
TEMPLATE_HASH = 'cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f'
HASHES = {
    'train.jsonl': 'f2851c4d747445d727e86fd03483bfce30250b0843db25b38f8a86d3a4aeaab3',
    'heldout.jsonl': '87fb5abbb936b05909856c6f22ec56fdc84d106b0d737aee3e0d49b60bcf1002',
}


def read_data(path):
    result = []
    with zipfile.ZipFile(path) as archive:
        for name, count in [('train.jsonl', 1400), ('heldout.jsonl', 100)]:
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != HASHES[name]:
                raise ValueError('Dataset hash mismatch: ' + name)
            rows = [json.loads(line) for line in raw.decode().splitlines()]
            if len(rows) != count or any(
                not all(isinstance(r.get(k), str) for k in ['instruction', 'input', 'output'])
                or not r['instruction'].strip() or not r['output'].strip() for r in rows
            ):
                raise ValueError('Invalid dataset schema or count: ' + name)
            result.append(rows)
    keys = [json.dumps(r, sort_keys=True) for rows in result for r in rows]
    if len(set(keys)) != 1500:
        raise ValueError('Duplicate or shared train/heldout row')
    return result


def encode(tok, row):
    user = row['instruction'] + ('\n\n' + row['input'] if row['input'] else '')
    messages = [{'role': 'user', 'content': user}]
    prompt = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template(messages + [{'role': 'assistant', 'content': row['output']}], tokenize=True)
    if full[:len(prompt)] != prompt or not len(prompt) < len(full) <= 512:
        raise ValueError('Template prefix, target length or 512-token limit failed')
    if tok.eos_token_id not in full[len(prompt):]:
        raise ValueError('Assistant EOS missing')
    return {'input_ids': full, 'attention_mask': [1] * len(full),
            'labels': [-100] * len(prompt) + full[len(prompt):]}
