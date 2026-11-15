"""Read-only CPU audit of retained results; does not evaluate a model."""
import datetime
import json
import re
from pathlib import Path
from experiment_data import read_data

root = Path(__file__).resolve().parents[1]
train, holdout = read_data(root / 'data/sampled_dataset.zip')
v = json.loads((root / 'evidence/verification.json').read_text())
gate = json.loads((root / 'evidence/ship_verdict_redacted.json').read_text())
assert gate['numerics'] == 'bfloat16' and gate['failed_rule'] == 'task_win'
assert gate['decision'] == "DON'T SHIP"
assert v['adapter_B_nonzero'] == v['adapter_B_count'] == 56
assert all(a['target_tokens'] == 457 for a in v['artifacts'].values())
assert len((root / 'data/ship_tasks.jsonl').read_text().splitlines()) == 6
peaks = {}
for stage, count, peak in [('train', 235, 5395), ('merge', 111, 115), ('quantize', 185, 1257)]:
    raw = (root / f'evidence/nvidia_smi_{stage}.txt').read_text()
    samples = [int(x) for x in re.findall(r'(\d+)MiB\s*/\s*\d+MiB', raw)]
    assert len(samples) == count and max(samples) == peak
    peaks[stage] = {'samples': count, 'peak_MiB': peak, 'peak_decimal_GB': peak * 2**20 / 1e9}
print(json.dumps({'audit_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Local file consistency only; no GPU execution', 'dataset_rows': [len(train), len(holdout)],
    'exact_duplicates_or_overlap': 0, 'dataset_hashes_match': True,
    'gpu_sample_peaks': peaks, 'gate_numerics': gate['numerics'], 'gate_input': 'base plus adapter',
    'nll_per_token': {k: a['heldout_nll_per_target_token'] for k, a in v['artifacts'].items()},
    'missing': ['NF4 gate', '100-row GPU results', 'safety cases', 'shuffled-label control', 'model weights']}, indent=2))
