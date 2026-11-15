"""Record every prompt, generation, stop reason and recomputable score."""
import argparse
import gc
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import utc
from variants import VARIANTS,load_variant,artifact_fingerprint
from eval.scorers import score
from eval.suites import load_suite


def validate_completed(path,items,meta,variant,suite,fingerprint):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows)!=len(items):raise ValueError('Incomplete saved evaluation')
    for row,item in zip(rows,items):
        if (row['item']!=item or row['id']!=item['id'] or row['prompt']!=item['prompt']
            or row['variant']!=variant or row['suite']!=suite
            or row['suite_sha256']!=meta['content_sha256'] or row['artifact_fingerprint']!=fingerprint
            or row['score']!=score(row['generation'],item)):
            raise ValueError('Saved evaluation identity or score changed')


def run_eval(run,variant,suite_dir,suite,out,shuffled=None,resume=False):
    import torch
    items,meta=load_suite(suite_dir,suite)
    fingerprint=artifact_fingerprint(variant,run,shuffled)
    if out.exists():
        if not resume:raise FileExistsError('Refusing to overwrite evaluation '+str(out))
        validate_completed(out,items,meta,variant,suite,fingerprint)
        return
    out.parent.mkdir(parents=True,exist_ok=True);partial=out.with_suffix('.partial.jsonl')
    if partial.exists():partial.rename(partial.with_name(partial.name+'.failed.'+str(__import__('time').time_ns())))
    model,tok=load_variant(variant,run,shuffled);outputs=0
    with partial.open('w') as f,torch.inference_mode():
        for item in items:
            ids=tok.apply_chat_template([{'role':'user','content':item['prompt']}],tokenize=True,return_dict=False,add_generation_prompt=True,return_tensors='pt').to(model.device)
            output=model.generate(ids,do_sample=False,max_new_tokens=meta.get('max_new_tokens',256),pad_token_id=tok.pad_token_id)
            new=output[0,ids.shape[1]:];answer=tok.decode(new,skip_special_tokens=True)
            row={'id':item['id'],'item':item,'prompt':item['prompt'],'generation':answer,'score':score(answer,item),
                 'finish_reason':'eos' if tok.eos_token_id in new.tolist() else 'length','variant':variant,
                 'suite':suite,'suite_sha256':meta['content_sha256'],'artifact_fingerprint':fingerprint,'utc':utc()}
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();outputs+=1
    if outputs!=len(items):raise ValueError('Incomplete case generation')
    partial.replace(out);del model;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--suite-dir',type=Path,required=True);p.add_argument('--variant',choices=VARIANTS);p.add_argument('--suite');p.add_argument('--out',type=Path);p.add_argument('--all-variants',action='store_true');p.add_argument('--shuffled-adapter',type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.all_variants:
        manifest=json.loads((a.suite_dir/'manifest.json').read_text())
        for name in VARIANTS:
            for suite in manifest['suites']:
                out=a.run_dir/'eval'/name/(suite+'.jsonl')
                run_eval(a.run_dir,name,a.suite_dir,suite,out,a.shuffled_adapter,a.resume)
    else:
        if not a.variant or not a.suite or not a.out:p.error('--variant, --suite and --out are required')
        run_eval(a.run_dir,a.variant,a.suite_dir,a.suite,a.out,a.shuffled_adapter,a.resume)


if __name__=='__main__':main()
