"""Fail-closed G1–G9 release decision on the saved NF4 artifact."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import ROOT,atomic_json,check_files,inventory,sha256,utc,valid_stage
from eval.scorers import score
from variants import artifact_fingerprint
from eval.stats import paired_bootstrap_diff,noninferior


def paired_cases(run,suite,a='merged_nf4',b='base_fp16',shuffled=None):
    def read(variant):
        rows=[json.loads(x) for x in (run/'eval'/variant/(suite+'.jsonl')).read_text().split('\n') if x]
        if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate cases')
        if any(r['variant']!=variant or r['suite']!=suite for r in rows):raise ValueError('Variant/suite mismatch')
        fingerprint=artifact_fingerprint(variant,run,shuffled)
        for row in rows:
            if row['artifact_fingerprint']!=fingerprint:raise ValueError('Evaluation artifact changed')
            measured=score(row['generation'],row['item'])
            if not math.isfinite(row['score']) or measured!=row['score'] or row['prompt']!=row['item']['prompt']:raise ValueError('Saved score/prompt mismatch')
        return {r['id']:r for r in rows}
    left,right=read(a),read(b)
    if left.keys()!=right.keys():raise ValueError('Paired case IDs differ')
    ids=sorted(left)
    for i in ids:
        if left[i]['item']!=right[i]['item'] or left[i]['suite_sha256']!=right[i]['suite_sha256']:raise ValueError('Paired content differs')
    return [left[i]['score'] for i in ids],[right[i]['score'] for i in ids],left


def compute_gate(run,criteria_path=ROOT/'configs/success_criteria.yaml',shuffled=None):
    run=Path(run);c=json.loads(Path(criteria_path).read_text());gates={};config={'n_boot':c['bootstrap_samples'],'seed':c['bootstrap_seed']}
    def record(name,fn):
        try:
            result=fn();gates[name]=result
        except Exception as error:gates[name]={'passed':False,'status':'MISSING OR INVALID EVIDENCE','reason':str(error)}
    def identity():
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from experiment_data import BASE,REVISION
        from rerun import model_dtype_kwargs,nf4_inventory
        manifest=json.loads((run/'ARTIFACTS.json').read_text());files=manifest['artifacts']['quantized']
        if not check_files(run/'quantized',files) or files!=inventory(run/'quantized'):raise ValueError('Artifact manifest mismatch')
        if not valid_stage(run,'quantize') or not valid_stage(run,'eval') or not valid_stage(run,'verify'):raise ValueError('Stage output hashes invalid')
        recorded=json.loads((run/'run_identity.json').read_text())
        if recorded['criteria_sha256']!=sha256(criteria_path):raise ValueError('Criteria changed after run registration')
        model=AutoModelForCausalLM.from_pretrained(run/'quantized',device_map={'':0},**model_dtype_kwargs(torch.float16));modules=nf4_inventory(model)
        tok=AutoTokenizer.from_pretrained(run/'quantized');base=AutoTokenizer.from_pretrained(BASE,revision=REVISION)
        equal=tok.get_vocab()==base.get_vocab() and tok.chat_template==base.chat_template and tok.eos_token_id==base.eos_token_id and tok.pad_token_id==base.pad_token_id
        del model;__import__('gc').collect();torch.cuda.empty_cache()
        return {'passed':equal,'files':files,'linear4bit_count':len(modules),'compute_dtype':'float16','tokenizer_equal':equal}
    record('G1',identity)
    def primary():
        a,b,_=paired_cases(run,'primary');ci=paired_bootstrap_diff(a,b,**config);base=sum(b)/len(b);s=c['primary']
        return dict(ci,base_accuracy=base,passed=len(a)>=s['min_n'] and s['base_min']<=base<=s['base_max'] and ci['low']>0 and ci['point']>=s['min_gain'])
    record('G2',primary)
    def safety():
        a,b,harm=paired_cases(run,'safety_harmful');s=c['safety_harmful'];harm_ci=noninferior(a,b,s['noninferiority_margin'],**config)
        x,y,benign=paired_cases(run,'safety_benign');over=paired_bootstrap_diff(x,y,alpha=.1,**config)
        manual=list(csv.DictReader((run/'manual_refusal_audit.csv').open()));known={('safety_harmful',i):r for i,r in harm.items()};known.update({('safety_benign',i):r for i,r in benign.items()})
        seen=set();agreement=[]
        for row in manual:
            key=(row['suite'],row['id'])
            if key in seen or key not in known or row['human_refusal'] not in ['0','1']:raise ValueError('Invalid/duplicate manual label')
            seen.add(key);saved=known[key]
            if row['generation_sha256']!=hashlib.sha256(saved['generation'].encode()).hexdigest() or not row['reviewer'].strip():raise ValueError('Manual audit not bound to generation/reviewer')
            agreement.append(int(row['human_refusal'])==saved['score'])
        rate=sum(agreement)/len(agreement) if agreement else 0
        passed=len(a)>=s['min_n'] and harm_ci['passed'] and harm_ci['point']>=s['min_point_delta'] and len(x)>=c['safety_benign']['min_n'] and over['high']<=c['safety_benign']['max_refusal_increase'] and len(manual)>=c['manual_refusal_audit']['min_n'] and rate>=c['manual_refusal_audit']['min_agreement']
        return {'passed':passed,'harmful':harm_ci,'benign_refusal_increase':over,'manual_n':len(manual),'manual_agreement':rate}
    record('G3',safety)
    def general():
        results={}
        for suite in c['general']['suites']:
            a,b,_=paired_cases(run,suite);ci=noninferior(a,b,c['general']['noninferiority_margin'],**config);ci['passed'] &= len(a)>=c['general']['min_n'];results[suite]=ci
        return {'passed':all(x['passed'] for x in results.values()),'suites':results}
    record('G4',general)
    def parity():
        results={}
        for suite,margin in [('primary',c['quantization']['max_primary_point_drop']),('safety_harmful',c['quantization']['max_safety_point_drop'])]:
            a,b,_=paired_cases(run,suite,b='merged_fp16');ci=paired_bootstrap_diff(a,b,**config);ci['passed']=ci['point']>=-margin;results[suite]=ci
        v=json.loads((run/'verification_extended.json').read_text());art=v['artifacts'];qa=art['merged_nf4'];mb=art['merged_fp16']
        if qa['executed_rows']!=100 or mb['executed_rows']!=100:raise ValueError('Missing 100-row NLL comparison')
        gap=qa['nll_per_token']-mb['nll_per_token']
        return {'passed':all(x['passed'] for x in results.values()) and math.isfinite(gap) and gap<=c['quantization']['max_nll_increase'],'comparisons':results,'nll_gap':gap,'nll_n':100}
    record('G5',parity)
    def controls():
        v=json.loads((run/'verification_extended.json').read_text());a,b,_=paired_cases(run,'primary',b='shuffled_label_adapter',shuffled=shuffled);ci=paired_bootstrap_diff(a,b,**config)
        flags=v['controls']
        return {'passed':flags['noop_logits_match'] and flags['wrong_scale_detected'] and v['adapter_merge_logits']['prompts']>=20 and ci['low']>0 and v['adapter_merge_logits']['top1_agreement']>=c['controls']['min_top1_agreement'],
                'noop_logits_match':flags['noop_logits_match'],'wrong_scale_detected':flags['wrong_scale_detected'],'shuffled_primary_diff':ci,'logit_prompts':v['adapter_merge_logits']['prompts']}
    record('G6',controls)
    def durability():
        r=json.loads((run/'remote_reload.json').read_text())
        return {'passed':r.get('passed') is True and r.get('fresh_process') is True and len(r['cases'])>=5 and r['artifact_manifest_sha256']==sha256(run/'ARTIFACTS.json'),'evidence':r}
    record('G7',durability)
    def command_check(command):
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        return {'passed':p.returncode==0,'exit_code':p.returncode,'output':p.stdout,'error':p.stderr}
    record('G8',lambda:command_check([sys.executable,'code/audit_evidence.py']))
    def hygiene():
        result=command_check([sys.executable,'tools/redact_check.py']);result['passed'] &= all((ROOT/n).is_file() for n in ['LICENSE','DATA_LICENSES.md','evidence/public_manifest.sha256'])
        return result
    record('G9',hygiene)
    failing=[name for name,result in gates.items() if result.get('passed') is not True]
    return {'created_utc':utc(),'decision':"DON'T SHIP" if failing else 'SHIP','criteria_sha256':sha256(criteria_path),'gates':gates,'failed_gates':failing,'artifact_directory':'quantized'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--shuffled-adapter',type=Path);a=p.parse_args()
    result=compute_gate(a.run_dir,shuffled=a.shuffled_adapter);atomic_json(a.out,result);print(result['decision']);print('Failing gates:',','.join(result['failed_gates']))
    # A negative decision is a completed gate, not an execution exception.


if __name__=='__main__':main()
