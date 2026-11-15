"""Select a primary task from saved, paired base baselines; never invent a target."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,atomic_json,sha256
from eval.stats import wilson_ci
from eval.scorers import score


def select(directory):
    candidates=[];selected=None
    for name in ['T-A','T-B','T-C']:
        path=Path(directory)/(name+'.jsonl')
        rows=[json.loads(x) for x in path.read_text().split('\n') if x]
        if len(rows)!=300 or len({x['id'] for x in rows})!=300:raise ValueError('Each candidate requires exactly 300 unique baseline cases')
        if any(x['variant']!='base_fp16' for x in rows):raise ValueError('Selection must use base_fp16 only')
        scores=[score(r['generation'],r['item']) for r in rows]
        if any(x not in [0,1] for x in scores):raise ValueError('Task selection requires binary exact/verifiable scores')
        total=int(sum(scores));accuracy=total/len(rows)
        candidate={'name':name,'n':len(rows),'correct':total,'accuracy':accuracy,'wilson_95':wilson_ci(total,len(rows)),'baseline_sha256':sha256(path)}
        candidates.append(candidate)
        if selected is None and .3<=accuracy<=.8:selected=name
    return {'status':'SELECTED' if selected else 'STOP: OWNER INPUT REQUIRED','selected':selected,'candidates':candidates,
            'rule':'first qualifying candidate T-A, T-B, T-C; baseline accuracy in [0.30,0.80]'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-dir',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'configs/task_selection.json');a=p.parse_args()
    result=select(a.baseline_dir);atomic_json(a.out,result)
    if result['selected'] is None:raise SystemExit('No task qualifies. Stop and ask the owner; do not alter thresholds.')


if __name__=='__main__':main()
