"""Fresh-process reload from persistent storage; run before the final release gate."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from common import ROOT,atomic_json,check_files,sha256,utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--local-copy',type=Path);a=p.parse_args()
    if a.local_copy:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from rerun import model_dtype_kwargs,nf4_inventory
        folder=a.local_copy/'quantized';manifest=json.loads((a.local_copy/'ARTIFACTS.json').read_text())
        if not check_files(folder,manifest['artifacts']['quantized']):raise RuntimeError('Remote artifact checksum mismatch')
        model=AutoModelForCausalLM.from_pretrained(folder,device_map={'':0},**model_dtype_kwargs(torch.float16));nf4_inventory(model)
        tok=AutoTokenizer.from_pretrained(folder)
        prompts=[json.loads(x)['prompt'] for x in (ROOT/'data/fixed_prompts.jsonl').read_text().split('\n') if x][:5]
        answers=[]
        for prompt in prompts:
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,return_dict=False,add_generation_prompt=True,return_tensors='pt').to(model.device)
            with torch.inference_mode():output=model.generate(ids,max_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id)
            answers.append({'prompt':prompt,'generation':tok.decode(output[0,ids.shape[1]:],skip_special_tokens=True)})
        if len(answers)!=5 or any(not x['generation'].strip() for x in answers):raise RuntimeError('Remote smoke failed')
        atomic_json(a.run_dir/'remote_reload.json',{'passed':True,'utc':utc(),'cases':answers,'artifact_manifest_sha256':sha256(a.local_copy/'ARTIFACTS.json'),'fresh_process':True});return
    from persist import artifact_manifest,save,restore
    artifact_manifest(a.run_dir);save(a.run_dir)
    with tempfile.TemporaryDirectory() as temp:
        copy=Path(temp)/a.run_dir.name
        if not restore(copy):raise RuntimeError('Remote copy missing')
        subprocess.run([sys.executable,__file__,'--run-dir',str(a.run_dir),'--local-copy',str(copy)],check=True)


if __name__=='__main__':main()
