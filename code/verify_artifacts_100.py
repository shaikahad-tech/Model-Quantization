import gc, json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

def tensor(folder, key):
    for file in Path(folder).glob('*.safetensors'):
        with safe_open(file, framework='pt', device='cpu') as sf:
            if key in sf.keys(): return sf.get_tensor(key).float()
    raise KeyError(key)

base='Qwen/Qwen2.5-1.5B-Instruct'
base_dir=snapshot_download(base, allow_patterns=['*.safetensors','*.json'])
keys=[f'model.layers.{i}.self_attn.{proj}.weight' for i in (0,13,27) for proj in ('q_proj','v_proj')]
deltas={k:((tensor('merged',k)-tensor(base_dir,k)).norm()/tensor(base_dir,k).norm()).item() for k in keys}
delta=min(deltas.values())
with safe_open('output/adapter_model.safetensors',framework='pt',device='cpu') as sf:
    bs=[sf.get_tensor(k).float().norm().item() for k in sf.keys() if 'lora_B' in k]
qcfg=json.loads(Path('quantized/config.json').read_text()).get('quantization_config',{})
rows=[json.loads(x) for x in Path('data/heldout.jsonl').read_text().splitlines()]
tok=AutoTokenizer.from_pretrained(base)
result={'base_revision':Path(base_dir).name,'sampled_weight_deltas':deltas,'min_sampled_delta':delta,'adapter_B_nonzero':sum(x>0 for x in bs),'adapter_B_count':len(bs),'merged_has_adapter_file':Path('merged/adapter_model.safetensors').exists(),'quantization_config':qcfg,'artifacts':{}}
for name,path in [('base',base),('merged','merged'),('4bit','quantized')]:
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,device_map='auto').eval()
    total=count=0; answers=[]; per_example=[]
    with torch.inference_mode():
        for row_index,row in enumerate(rows):
            user=row['instruction']+('\n'+row['input'] if row.get('input') else '')
            prompt=tok.apply_chat_template([{'role':'user','content':user}],tokenize=True,add_generation_prompt=True)['input_ids']
            full=tok.apply_chat_template([{'role':'user','content':user},{'role':'assistant','content':row['output']}],tokenize=True)['input_ids']
            assert full[:len(prompt)]==prompt and len(prompt)<len(full)<=512
            ids=torch.tensor([full],device=model.device)
            logits=model(ids).logits[:,len(prompt)-1:-1].float()
            target=ids[:,len(prompt):]
            loss_sum=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),reduction='sum').item()
            total+=loss_sum; count+=target.numel()
            per_example.append({'index':row_index,'nll_per_token':loss_sum/target.numel(),'target_tokens':target.numel()})
        for question in ['Why is a bicycle helmet useful?','List three steps for washing hands.']:
            ids=tok.apply_chat_template([{'role':'user','content':question}],tokenize=True,add_generation_prompt=True,return_tensors='pt')['input_ids'].to(model.device)
            answer=model.generate(ids,do_sample=False,max_new_tokens=64,pad_token_id=tok.eos_token_id)
            answers.append(tok.decode(answer[0,ids.shape[1]:],skip_special_tokens=True))
    result['artifacts'][name]={'heldout_nll_per_target_token':total/count,'target_tokens':count,'per_example':per_example,'answers':answers}
    print(name,json.dumps(result['artifacts'][name]),flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
result['structural_pass']=delta>1e-6 and len(bs)>0 and all(x>0 for x in bs) and not result['merged_has_adapter_file']
assert len(rows)==100 and all(len(result['artifacts'][n]['per_example'])==100 for n in result['artifacts'])
result['quantized_close_nll']=abs(result['artifacts']['4bit']['heldout_nll_per_target_token']-result['artifacts']['merged']['heldout_nll_per_target_token'])<0.1
Path('verification_100.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert result['structural_pass'] and result['quantized_close_nll']
