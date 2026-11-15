"""CPU checks for data integrity, template boundary and assistant-only masking."""
import unittest
import tempfile
import zipfile
from pathlib import Path
from experiment_data import encode, read_data


class TokenizerFixture:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        prompt = [1, 2, 3]
        if add_generation_prompt:
            return prompt
        return prompt + [4] * len(messages[-1]['content']) + [99]


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.row = {'instruction': 'Question', 'input': '', 'output': 'Answer'}

    def test_original_dataset(self):
        train, heldout = read_data(Path(__file__).resolve().parents[1] / 'data/sampled_dataset.zip')
        self.assertEqual((len(train), len(heldout)), (1400, 100))

    def test_prompt_mask_and_eos(self):
        item = encode(TokenizerFixture(), self.row)
        self.assertEqual(item['labels'][:3], [-100] * 3)
        self.assertEqual(item['labels'][3:], item['input_ids'][3:])
        self.assertEqual(item['labels'][-1], 99)
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


if __name__ == '__main__':
    unittest.main()
