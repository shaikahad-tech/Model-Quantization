"""Config-driven stages; all GPU execution must be recorded and persisted."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT,atomic_json,check_files,complete_stage,inventory,load_config,sha256,utc,valid_stage
from experiment_data import BASE,REVISION,TEMPLATE_HASH,encode,jsonl,mask_audit,read_data
from budget import calculate


def model_dtype_kwargs(dtype):
    major=int(importlib.metadata.version('transformers').split('.')[0])
    return {'torch_dtype' if major<5 else 'dtype':dtype}


def dtype_census(model):
    result={}
    for name,p in model.named_parameters():
        key=str(p.dtype);result[key]=result.get(key,0)+p.numel()
    return {'parameters_by_storage_dtype':result,'embedding_dtype':str(model.get_input_embeddings().weight.dtype)}


def nf4_inventory(model):
    import bitsandbytes as bnb
    import torch
    modules={n:str(m.compute_dtype) for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear4bit)}
    expected={f'model.layers.{i}.{part}.{proj}' for i in range(28)
              for part,projs in [('self_attn',['q_proj','k_proj','v_proj','o_proj']),('mlp',['gate_proj','up_proj','down_proj'])] for proj in projs}
    if set(modules)!=expected or any(v!=str(torch.float16) for v in modules.values()):
        raise ValueError('Saved NF4 module coverage or compute dtype mismatch')
    return modules


def same_mask_validation(model,features):
    import torch
    case_losses=[];summed=0.;tokens=0
    model.eval()
    with torch.inference_mode():
        for item in features:
            ids=torch.tensor([item['input_ids']],device=model.device)
            labels=torch.tensor([item['labels']],device=model.device)
            out=model(input_ids=ids,labels=labels)
            count=int((labels[:,1:]!=-100).sum())
            case_losses.append(float(out.loss));summed+=float(out.loss)*count;tokens+=count
    return {'rows':len(features),'target_tokens':tokens,'mean_example_loss':sum(case_losses)/len(case_losses),
            'token_weighted_nll':summed/tokens,'per_example_loss':case_losses}


def execute(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training
    from persist import save,artifact_manifest
    run=args.run_dir.resolve();c=load_config(args.config)
    if args.seed is not None:c['seed']=args.seed
    if c['model']!=BASE or c['revision']!=REVISION:raise ValueError('This pipeline is calibrated only for the pinned Qwen model')
    set_seed(c['seed']);tok=AutoTokenizer.from_pretrained(BASE,revision=REVISION)
    if hashlib.sha256(tok.chat_template.encode()).hexdigest()!=TEMPLATE_HASH:raise ValueError('Template hash mismatch')
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    dtype=model_dtype_kwargs(torch.float16);paths={n:run/n for n in ['adapter','merged','quantized']}
    stage=args.stage;outputs=[]
    if stage=='train':
        from datasets import Dataset
        from transformers import Trainer,TrainerCallback,TrainingArguments,DataCollatorForSeq2Seq
        from transformers.trainer_utils import get_last_checkpoint
        if c['dataset']=='historical':
            train,holdout=read_data(ROOT/'data/sampled_dataset.zip');validation=train[1260:];train=train[:1260]
        else:
            folder=ROOT/c['dataset'];train=jsonl((folder/'train.jsonl').read_bytes());validation=jsonl((folder/'val.jsonl').read_bytes());holdout=[]
            if not (folder/'manifest.json').is_file():raise RuntimeError('v2 dataset has not been built or licensed')
        if args.shuffled_labels:
            answers=[r['output'] for r in train];train=[dict(r,output=answers[(i+1)%len(answers)]) for i,r in enumerate(train)]
        kwargs={'max_len':c['max_length'],'supervise_trailing_newline':c['supervise_trailing_newline']}
        audit=mask_audit(tok,train+validation+holdout,**kwargs);atomic_json(run/'mask_audit.json',audit)
        if audit['zero_target_rows'] or audit['role_token_rows'] or (not c['supervise_trailing_newline'] and audit['eos_not_last_rows']):raise ValueError('Mask audit failed')
        features=[encode(tok,r,**kwargs) for r in train];val=[encode(tok,r,**kwargs) for r in validation]
        model=AutoModelForCausalLM.from_pretrained(BASE,revision=REVISION,quantization_config=q,device_map={'':0},**dtype)
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
        census=dtype_census(model)
        model=get_peft_model(model,LoraConfig(r=c['r'],lora_alpha=c['alpha'],target_modules=c['targets'],lora_dropout=c['dropout'],bias='none',task_type='CAUSAL_LM'))
        model.config.use_cache=False
        actual=sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected=calculate(c)['trainable_parameters']
        if actual!=expected:raise ValueError(f'Trainable parameter mismatch {actual} != {expected}')
        atomic_json(run/'preflight.json',{'config':c,'config_sha256':sha256(args.config),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','tokenizers']},
            'dtype_after_prepare':census,'dtype_with_adapter':dtype_census(model),'trainable_parameters':actual,
            'template_sha256':TEMPLATE_HASH,'data_sha256':hashlib.sha256(json.dumps(train+validation,sort_keys=True).encode()).hexdigest(),
            'effective_batch':c['batch_size']*c['accumulation'],'shuffled_label_control':args.shuffled_labels})
        options=TrainingArguments(output_dir=str(run/'checkpoints'),num_train_epochs=c['epochs'],
            per_device_train_batch_size=c['batch_size'],per_device_eval_batch_size=1,gradient_accumulation_steps=c['accumulation'],
            learning_rate=c['lr'],lr_scheduler_type=c['scheduler'],warmup_ratio=c['warmup_ratio'],
            fp16=True,gradient_checkpointing=True,logging_steps=10,eval_strategy='steps',eval_steps=c['checkpoint_steps'],
            save_strategy='steps',save_steps=c['checkpoint_steps'],save_total_limit=3,load_best_model_at_end=True,
            metric_for_best_model='eval_selection_score' if c['name']=='v2' else 'eval_loss',greater_is_better=c['name']=='v2',report_to='none',seed=c['seed'],data_seed=c['seed'],optim=c['optimizer'],weight_decay=0.0)
        class PersistCheckpoint(TrainerCallback):
            def on_save(self,args,state,control,**kwargs):
                checkpoint=f'checkpoints/checkpoint-{state.global_step}'
                atomic_json(run/checkpoint/'CHECKPOINT_HASHES.json',[r for r in inventory(run/checkpoint) if r['path']!='CHECKPOINT_HASHES.json'])
                save(run)
        trainer_type=Trainer
        if c['name']=='v2':
            from dev_selection import trainer_class
            trainer_type=trainer_class(Trainer,tok,ROOT/c['dev_suite_dir'],run)
        trainer=trainer_type(model=model,args=options,train_dataset=Dataset.from_list(features),eval_dataset=Dataset.from_list(val),
            data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100,pad_to_multiple_of=8),callbacks=[PersistCheckpoint()])
        checkpoint=get_last_checkpoint(str(run/'checkpoints')) if args.resume and (run/'checkpoints').exists() else None
        if checkpoint:
            recorded=Path(checkpoint)/'CHECKPOINT_HASHES.json'
            if not recorded.exists() or not check_files(checkpoint,json.loads(recorded.read_text())):raise ValueError('Incomplete/corrupt persisted checkpoint')
        if not (run/'eval_baseline.json').exists():
            if checkpoint:raise RuntimeError('Cannot reconstruct a missing baseline from a trained checkpoint')
            baseline=trainer.evaluate();baseline['same_mask']=same_mask_validation(model,val);atomic_json(run/'eval_baseline.json',baseline);save(run)
        trainer.train(resume_from_checkpoint=checkpoint)
        final=trainer.evaluate();final['same_mask']=same_mask_validation(model,val);atomic_json(run/'eval_final.json',final)
        # Keep unsafe candidates and negative controls; selection and release gates reject them.
        trainer.save_model(str(paths['adapter']));tok.save_pretrained(paths['adapter'])
        outputs=[run/n for n in ['mask_audit.json','preflight.json','eval_baseline.json','eval_final.json']]+list(paths['adapter'].rglob('*'))+list(run.glob('dev_*.json'))
    elif stage=='merge':
        model=AutoModelForCausalLM.from_pretrained(BASE,revision=REVISION,device_map={'':'cpu'},**dtype)
        model=PeftModel.from_pretrained(model,paths['adapter']).merge_and_unload(safe_merge=True)
        model.save_pretrained(paths['merged'],safe_serialization=True);tok.save_pretrained(paths['merged']);outputs=list(paths['merged'].rglob('*'))
    elif stage=='quantize':
        model=AutoModelForCausalLM.from_pretrained(paths['merged'],quantization_config=q,device_map={'':0},**dtype)
        model.save_pretrained(paths['quantized'],safe_serialization=True);tok.save_pretrained(paths['quantized'])
        del model;gc.collect();torch.cuda.empty_cache()
        model=AutoModelForCausalLM.from_pretrained(paths['quantized'],device_map={'':0},**dtype)
        modules=nf4_inventory(model);size=sum(p.stat().st_size for p in paths['quantized'].glob('*.safetensors'))
        predicted=calculate(c)['nf4_analytic_bytes']
        if abs(size-predicted)/predicted>.01:raise ValueError('NF4 serialization differs by more than 1% from analytic estimate')
        atomic_json(run/'quantize_checks.json',{'modules':modules,'weight_bytes':size,'analytic_bytes':predicted});outputs=list(paths['quantized'].rglob('*'))+[run/'quantize_checks.json']
    elif stage=='verify':
        from verify_extended import verify
        verify(run,tok,shuffled=args.shuffled_adapter)
        outputs=[run/'verification_extended.json',run/'fixed_prompt_logits.pt']
    else:raise ValueError('Unsupported GPU stage')
    if stage in ['train','merge','quantize']:artifact_manifest(run)
    return [p for p in outputs if p.is_file()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['train','merge','quantize','verify']);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/train_replica.yaml');p.add_argument('--seed',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--shuffled-labels',action='store_true');p.add_argument('--shuffled-adapter',type=Path)
    a=p.parse_args();run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    if not (run/'logs/budget_before_run.json').exists() or not (run/'run_identity.json').exists():raise SystemExit('Start via reproduce.sh so the budget/config are recorded before training')
    if a.resume and valid_stage(run,a.stage):print('Verified completed stage:',a.stage);return
    dependencies={'merge':'train','quantize':'merge','verify':'quantize'}
    if a.stage in dependencies and not valid_stage(run,dependencies[a.stage]):raise SystemExit('Missing or invalid prerequisite stage')
    from persist import destination,save
    destination()
    import torch
    if not torch.cuda.is_available():raise SystemExit('CUDA GPU required')
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        files=execute(a);torch.cuda.synchronize()
        metrics={'max_memory_allocated':torch.cuda.max_memory_allocated(),'max_memory_reserved':torch.cuda.max_memory_reserved(),
                 'wall_seconds':time.monotonic()-start,'host_ru_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'config_sha256':sha256(a.config)}
        complete_stage(run,a.stage,files,metrics);save(run)
    except Exception as error:
        atomic_json(run/f'failure_{a.stage}_{time.time_ns()}.json',{'utc':utc(),'stage':a.stage,'error':str(error),'wall_seconds':time.monotonic()-start})
        try:save(run)
        except Exception as persistence_error:print('Failure persistence also failed:',persistence_error,file=sys.stderr)
        raise


if __name__=='__main__':main()
