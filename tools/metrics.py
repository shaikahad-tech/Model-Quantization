"""Compute public numeric facts from retained evidence, never rewrite source evidence."""
import datetime
import hashlib
import json
import re
import statistics
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from budget import calculate
from common import load_config
from experiment_data import read_data,dedupe_report


def compute():
    train,heldout=read_data(ROOT/'data/sampled_dataset.zip');d=dedupe_report(train,heldout)
    result={'train_rows':len(train),'heldout_rows':len(heldout),'exact_overlap':d['exact_overlap'],'duplicate_instructions':d['duplicate_train_instructions'],'near_pairs':len(d['near_instruction_pairs'])}
    for stage in ['train','merge','quantize']:
        raw=(ROOT/f'evidence/nvidia_smi_{stage}.txt').read_text()
        values=[int(x) for x in re.findall(r'(\d+)MiB\s*/\s*\d+MiB',raw)]
        times=[datetime.datetime.fromisoformat(x) for x in re.findall(r'^\d{4}-\d\d-\d\dT[^\n]+',raw,re.M)]
        result[stage+'_peak_mib']=max(values);result[stage+'_samples']=len(values);result[stage+'_peak_gb']=max(values)*2**20/1e9
        result[stage+'_interval_seconds']=statistics.mean((b-a).total_seconds() for a,b in zip(times,times[1:]))
    v=json.loads((ROOT/'evidence/verification.json').read_text());text=(ROOT/'evidence/initial_verification_output_redacted.txt').read_text()
    start=text.index('{\n  "base_revision"');v1,_=json.JSONDecoder().raw_decode(text[start:])
    for name,variant in [('base','base'),('merged','merged'),('nf4','4bit')]:
        result[name+'_nll']=v['artifacts'][variant]['heldout_nll_per_target_token'];result['vm1_'+name+'_nll']=v1['artifacts'][variant]['heldout_nll_per_target_token']
    result['q_delta']=v['relative_q_proj_delta'];result['vm1_q_delta']=v1['relative_q_proj_delta'];result['target_tokens']=v['artifacts']['base']['target_tokens'];result['adapter_B_nonzero']=v['adapter_B_nonzero']
    for vm,file in [('vm1','initial_train_excerpt_redacted.txt'),('vm2','train_log_redacted.txt')]:
        log=(ROOT/'evidence'/file).read_text();result[vm+'_val_loss']=float(re.search(r'Val loss: ([\d.]+)',log)[1]);peak,total=re.search(r'GPU peak: ([\d.]+)/([\d.]+) GB',log).groups();result[vm+'_cli_peak_gb']=float(peak);result[vm+'_cli_total_gb']=float(total)
    result.update(calculate(load_config(ROOT/'configs/train_replica.yaml')))
    gate=json.loads((ROOT/'evidence/ship_verdict_redacted.json').read_text())
    result['task_base']=gate['task_win']['base'];result['task_tuned']=gate['task_win']['tuned']
    for row in gate['benchmark_deltas']:
        result[row['name']+'_base']=row['base'];result[row['name']+'_tuned']=row['tuned']
    result['gate_numerics']=gate['numerics'];result['task_items']=len((ROOT/'data/ship_tasks.jsonl').read_text().strip().split('\n'))
    return result


def render(text,metrics):
    def replace(match):
        key,spec=match.group(1),match.group(2)
        return format(metrics[key],spec or '')
    return re.sub(r'\{\{([A-Za-z0-9_]+)(?::([^}]+))?\}\}',replace,text)


if __name__=='__main__':print(json.dumps(compute(),indent=2))
