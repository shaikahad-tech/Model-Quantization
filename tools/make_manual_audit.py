"""Create a blank human-label sheet; never fills human decisions automatically."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();out=a.run_dir/'manual_refusal_audit.csv'
    if out.exists():raise SystemExit('Do not overwrite human labels')
    selected=[];rng=random.Random(20261111)
    for suite in ['safety_harmful','safety_benign']:
        rows=[json.loads(x) for x in (a.run_dir/'eval/merged_nf4'/(suite+'.jsonl')).read_text().split('\n') if x]
        if len(rows)<25:raise ValueError('Need at least 25 cases from each safety suite')
        for row in rng.sample(rows,25):selected.append({'suite':suite,'id':row['id'],'prompt':row['prompt'],'generation':row['generation'],'generation_sha256':hashlib.sha256(row['generation'].encode()).hexdigest(),'human_refusal':'','reviewer':''})
    with out.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)


if __name__=='__main__':main()
