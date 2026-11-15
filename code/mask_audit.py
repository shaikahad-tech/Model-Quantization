import argparse
import json
from pathlib import Path
from experiment_data import BASE, REVISION, jsonl, mask_audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-length',type=int,default=512)
    a=p.parse_args()
    from transformers import AutoTokenizer
    result=mask_audit(AutoTokenizer.from_pretrained(BASE,revision=REVISION),jsonl(a.data.read_bytes()),a.max_length)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    if result['zero_target_rows'] or result['role_token_rows'] or result['eos_not_last_rows']:
        raise SystemExit('Mask audit failed; per-row failures preserved')


if __name__=='__main__':main()
