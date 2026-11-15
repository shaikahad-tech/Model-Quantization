"""Saved-artifact diagnostics. A passing diagnostic is not a release decision."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,atomic_json,inventory
from experiment_data import encode,read_data
from model_checks import calibrate_tiny_merge,merge_residual
from variants import VARIANTS,load_variant
from eval.stats import paired_bootstrap_diff


def nll_cases(model,tok,rows,historical=False):
    import torch
    cases=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            if historical:
                user=row['instruction']+('\n'+row['input'] if row['input'] else '')
                messages=[{'role':'user','content':user}]
                prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
                full=tok.apply_chat_template(messages+[{'role':'assistant','content':row['output']}],tokenize=True,return_dict=False)
                if full[:len(prompt)]!=prompt or not len(prompt)<len(full)<=512:raise ValueError('Historical formatter prefix/length failure')
                item={'input_ids':full,'labels':[-100]*len(prompt)+full[len(prompt):]}
            else:item=encode(tok,row)
            ids=torch.tensor([item['input_ids']],device=model.device);labels=torch.tensor([item['labels']],device=model.device)
            logits=model(ids).logits[:,:-1].float();targets=labels[:,1:]
            total=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='sum')
            count=int((targets!=-100).sum());value=float(total)
            if not torch.isfinite(total) or count==0:raise ValueError('Invalid NLL/denominator')
            cases.append({'id':f'dolly-{i:03d}','index':i,'target_tokens':count,'nll_sum':value,'nll_per_token':value/count})
    return cases


def verify(run,tok,shuffled=None):
    import torch
    from safetensors import safe_open
    from rerun import dtype_census,nf4_inventory
    from eval.scorers import contains
    run=Path(run);_,rows=read_data(ROOT/'data/sampled_dataset.zip')
    if len(rows)!=100:raise ValueError('Expected exactly 100 heldouts')
    prompts=[json.loads(x) for x in (ROOT/'data/fixed_prompts.jsonl').read_text().split('\n') if x]
    if len(prompts)!=20:raise ValueError('Expected 20 fixed prompts')
    tasks=[json.loads(x) for x in (ROOT/'data/ship_tasks.jsonl').read_text().split('\n') if x]
    result={'artifacts':{},'controls':{},'status':'INCOMPLETE','safety':'NOT RUN by this diagnostic','merge_calibration':calibrate_tiny_merge()}
    weights={}
    with safe_open(run/'adapter/adapter_model.safetensors',framework='pt',device='cpu') as f:weights={k:f.get_tensor(k).float() for k in f.keys()}
    modules=[k.removeprefix('base_model.model.').removesuffix('.lora_A.weight') for k in weights if k.endswith('.lora_A.weight')]
    cfg=json.loads((run/'adapter/adapter_config.json').read_text())
    if cfg.get('use_dora') or cfg.get('use_rslora') or cfg.get('rank_pattern') or cfg.get('alpha_pattern') or cfg.get('modules_to_save') or cfg.get('bias')!='none':raise ValueError('Unsupported merge variant')
    originals={};logits_by_variant={};variants=[x for x in VARIANTS if x!='shuffled_label_adapter' or shuffled or (run/'shuffled_label_adapter').exists()]
    for name in variants:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model,variant_tok=load_variant(name,run,shuffled)
        if variant_tok.get_vocab()!=tok.get_vocab() or variant_tok.chat_template!=tok.chat_template or variant_tok.eos_token_id!=tok.eos_token_id or variant_tok.pad_token_id!=tok.pad_token_id:raise ValueError('Tokenizer identity mismatch')
        if name=='base_fp16':originals={m:model.get_submodule(m).weight.detach().float().cpu() for m in modules}
        if name=='merged_fp16':
            result['merge_checks']=[]
            for m in modules:
                r=merge_residual(originals[m],model.get_submodule(m).weight.detach(),weights['base_model.model.'+m+'.lora_A.weight'],weights['base_model.model.'+m+'.lora_B.weight'],cfg['lora_alpha']/cfg['r'])
                result['merge_checks'].append(dict(r,module=m,passed=r['max_ulp']<=result['merge_calibration']['tolerance_ulp']))
        if name=='merged_nf4':result['nf4_modules']=nf4_inventory(model)
        cases=nll_cases(model,tok,rows);historic=nll_cases(model,tok,rows[:8],historical=True);fixed=[];task_results=[]
        with torch.inference_mode():
            for item in prompts:
                ids=tok.apply_chat_template([{'role':'user','content':item['prompt']}],tokenize=True,return_dict=False,add_generation_prompt=True,return_tensors='pt').to(model.device)
                fixed.append(model(ids).logits[0,-1].float().cpu())
            for i,item in enumerate(tasks):
                ids=tok.apply_chat_template([{'role':'user','content':item['prompt']}],tokenize=True,return_dict=False,add_generation_prompt=True,return_tensors='pt').to(model.device)
                out=model.generate(ids,max_new_tokens=256,do_sample=False,pad_token_id=tok.pad_token_id);new=out[0,ids.shape[1]:]
                answer=tok.decode(new,skip_special_tokens=True)
                task_results.append({'id':i,'prompt':item['prompt'],'expected':item['expected'],'generation':answer,
                    'score':contains(answer,item['expected']),'contains_case_sensitive':item['expected'] in answer,
                    'finish_reason':'eos' if tok.eos_token_id in new.tolist() else 'length'})
        logits_by_variant[name]=torch.stack(fixed)
        artifact_path=run/('quantized' if name=='merged_nf4' else 'merged' if name=='merged_fp16' else 'adapter')
        result['artifacts'][name]={'executed_rows':len(cases),'cases':cases,'nll_per_token':sum(c['nll_sum'] for c in cases)/sum(c['target_tokens'] for c in cases),
            'historical_eight_row_nll':sum(c['nll_sum'] for c in historic)/sum(c['target_tokens'] for c in historic),'historical_eight_row_tokens':sum(c['target_tokens'] for c in historic),
            'tasks':task_results,'dtype_census':dtype_census(model),'tokenizer_equal':True,'max_memory_allocated':torch.cuda.max_memory_allocated(),
            'files':inventory(artifact_path) if name in ['merged_fp16','merged_nf4','fp16_adapter','nf4_adapter'] else [],'base_revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306'}
        atomic_json(run/'verification_extended.partial.json',result);del model;gc.collect();torch.cuda.empty_cache()
    for name,artifact in result['artifacts'].items():
        values=[c['nll_per_token'] for c in artifact['cases']]
        artifact['nll_diff_ci']={ref:paired_bootstrap_diff(values,[c['nll_per_token'] for c in result['artifacts'][ref]['cases']]) for ref in ['base_fp16','merged_fp16']}
    base=logits_by_variant['base_fp16'];noop=logits_by_variant['noop'];left=logits_by_variant['fp16_adapter'];right=logits_by_variant['merged_fp16']
    result['controls']['noop_logits_match']=torch.allclose(base,noop,atol=1e-5,rtol=1e-5)
    result['controls']['wrong_scale_detected']=result['merge_calibration']['wrong_scale_detected']
    result['adapter_merge_logits']={'max_abs':float((left-right).abs().max()),'relative_l2':float((left-right).norm()/left.norm()),'top1_agreement':float((left.argmax(-1)==right.argmax(-1)).float().mean()),'prompts':len(prompts)}
    controls=['noop','randomized_B','shuffled_label_adapter'];ci={}
    for name in controls:
        if name in result['artifacts']:
            ci[name]=paired_bootstrap_diff([c['nll_per_token'] for c in result['artifacts'][name]['cases']],[c['nll_per_token'] for c in result['artifacts']['fp16_adapter']['cases']])
    result['controls']['nll_advantage_ci']=ci
    result['training_signal_pass']=len(ci)==3 and all(c['low']>0 for c in ci.values())
    result['status']='DIAGNOSTICS COMPLETED; RELEASE GATE SEPARATE'
    result['limitations']='NLL measures reference fit; S1 task superiority, safety, human refusal audit and remote reload are checked separately. Missing shuffled-label adapter prevents a positive training-signal finding.'
    torch.save(logits_by_variant,run/'fixed_prompt_logits.pt');atomic_json(run/'verification_extended.json',result)
    if not result['controls']['noop_logits_match'] or not all(c['passed'] for c in result['merge_checks']) or result['adapter_merge_logits']['top1_agreement']<.999:raise ValueError('Structural/logit checks failed; results preserved')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--shuffled-adapter',type=Path);a=p.parse_args()
    from transformers import AutoTokenizer
    from experiment_data import BASE,REVISION
    verify(a.run_dir,AutoTokenizer.from_pretrained(BASE,revision=REVISION),a.shuffled_adapter)


if __name__=='__main__':main()
