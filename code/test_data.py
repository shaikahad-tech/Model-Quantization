"""CPU checks for data integrity, template boundary and assistant-only masking."""
import unittest
import tempfile
import zipfile
from pathlib import Path
from experiment_data import encode, read_data, dedupe_report, mask_audit, BASE, REVISION, TEMPLATE_HASH


class TokenizerFixture:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=True):
        if return_dict:
            return {"input_ids": [1, 2, 3]}
        prompt = [1, 2, 3]
        if add_generation_prompt:
            return prompt
        return prompt + [4] * len(messages[-1]['content']) + [99, 10]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join('<|im_end|>' if x==99 else 'x' for x in ids)


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.row = {'instruction': 'Question', 'input': '', 'output': 'Answer'}

    def test_original_dataset(self):
        train, heldout = read_data(Path(__file__).resolve().parents[1] / 'data/sampled_dataset.zip')
        self.assertEqual((len(train), len(heldout)), (1400, 100))

    def test_prompt_mask_and_eos(self):
        item = encode(TokenizerFixture(), self.row)
        self.assertEqual(item['labels'][:3], [-100] * 3)
        self.assertEqual(item['labels'][3:-1], item['input_ids'][3:-1])
        self.assertEqual(item['labels'][-1], -100)
        self.assertEqual([x for x in item['labels'] if x!=-100][-1],99)
        self.assertEqual(item['attention_mask'], [1] * len(item['input_ids']))

    def test_reject_changed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('train.jsonl', '{}\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                read_data(path)

    def test_context_separator(self):
        class Capture(TokenizerFixture):
            def apply_chat_template(inner, messages, **kwargs):
                self.assertEqual(messages[0]['content'], 'Question\n\nContext')
                return super().apply_chat_template(messages, **kwargs)
        encode(Capture(), dict(self.row, input='Context'))

    def test_reject_truncation(self):
        with self.assertRaisesRegex(ValueError, '512-token'):
            encode(TokenizerFixture(), dict(self.row, output='x' * 512))

    def test_reject_missing_eos(self):
        class BadEOS(TokenizerFixture):
            eos_token_id = 999
        with self.assertRaisesRegex(ValueError, 'EOS'):
            encode(BadEOS(), self.row)

    def test_reject_prefix_mismatch(self):
        class BadPrefix(TokenizerFixture):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                return [9] + ids if kwargs.get('add_generation_prompt') else ids
        with self.assertRaisesRegex(ValueError, 'prefix'):
            encode(BadPrefix(), self.row)


    def test_unicode_jsonl(self):
        import hashlib, json
        raw=(json.dumps(dict(self.row, output='a\u2028b\u0085c'),ensure_ascii=False)+'\n').encode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unicode.zip'
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('rows.jsonl',raw)
            result=read_data(path,{'rows.jsonl':hashlib.sha256(raw).hexdigest()},{'rows.jsonl':1})
            self.assertEqual(result[0][0]['output'],'a\u2028b\u0085c')

    def test_near_duplicates(self):
        report=dedupe_report([self.row,self.row],[dict(self.row,instruction='Question')])
        self.assertEqual(report['exact_overlap'],1)
        self.assertEqual(report['duplicate_train_instructions'],1)
        self.assertEqual(len(report['near_instruction_pairs']),2)

    def test_mask_zero_target(self):
        result=mask_audit(TokenizerFixture(),[dict(self.row,output='x'*600)])
        self.assertEqual(result['zero_target_rows'],1)

    def test_return_dict_false_contract(self):
        self.assertIsInstance(TokenizerFixture().apply_chat_template([]),dict)
        self.assertIsInstance(encode(TokenizerFixture(),self.row)['input_ids'],list)

    def test_real_tokenizer(self):
        import hashlib
        try:
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(BASE,revision=REVISION)
        except (ImportError,OSError) as error:
            self.skipTest('Tokenizer unavailable: '+str(type(error).__name__))
        self.assertEqual(hashlib.sha256(tok.chat_template.encode()).hexdigest(),TEMPLATE_HASH)
        train,_=read_data(Path(__file__).resolve().parents[1]/'data/sampled_dataset.zip')
        for row in train[:3]:
            item=encode(tok,row)
            supervised=[i for i in item['labels'] if i!=-100]
            self.assertEqual(tok.decode(supervised,skip_special_tokens=False),row['output']+'<|im_end|>')


if __name__ == '__main__':
    unittest.main()
