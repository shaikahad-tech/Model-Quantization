"""Explicit model identity: NF4 adapter is distinct from FP16 adapter."""
from pathlib import Path
from experiment_data import BASE,REVISION
from rerun import model_dtype_kwargs

VARIANTS=['base_fp16','base_nf4','fp16_adapter','nf4_adapter','merged_fp16','merged_nf4','shuffled_label_adapter','noop','randomized_B']


def load_variant(name,run,shuffled=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel,LoraConfig,get_peft_model
    from model_checks import randomize_b
    if name not in VARIANTS:raise ValueError('Unknown variant')
    run=Path(run);path=run/('merged' if name=='merged_fp16' else 'quantized') if name in ['merged_fp16','merged_nf4'] else BASE
    kwargs={'revision':REVISION} if path==BASE else {}
    if name in ['base_nf4','nf4_adapter']:
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(path,device_map={'':0},**model_dtype_kwargs(torch.float16),**kwargs)
    tok=AutoTokenizer.from_pretrained(path,**({'revision':REVISION} if path==BASE else {}))
    if name in ['fp16_adapter','nf4_adapter','randomized_B','shuffled_label_adapter']:
        adapter=Path(shuffled) if shuffled else run/'shuffled_label_adapter'
        if name!='shuffled_label_adapter':adapter=run/'adapter'
        if not adapter.is_dir():raise FileNotFoundError('Missing '+name+' adapter: '+str(adapter))
        model=PeftModel.from_pretrained(model,adapter)
        if name=='randomized_B':randomize_b(model)
    elif name=='noop':
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    return model.eval(),tok


def artifact_fingerprint(name,run,shuffled=None):
    import hashlib,json
    from common import inventory
    if name not in VARIANTS:raise ValueError('Unknown variant')
    if name in ['base_fp16','base_nf4','noop']:
        identity=[{'base_revision':REVISION}]
    else:
        folder={'merged_nf4':'quantized','merged_fp16':'merged','shuffled_label_adapter':'shuffled_label_adapter'}.get(name,'adapter')
        path=Path(shuffled) if name=='shuffled_label_adapter' and shuffled else Path(run)/folder
        if not path.is_dir():raise FileNotFoundError(path)
        identity=inventory(path)
        if not identity:raise ValueError('Empty artifact directory')
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
