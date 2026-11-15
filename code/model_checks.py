"""Model invariants shared by CPU controls and GPU verification."""
import copy


def randomize_b(model,seed=20261111):
    import torch
    generator=torch.Generator(device='cpu').manual_seed(seed);changed=[]
    with torch.no_grad():
        for name,p in model.named_parameters():
            if '.lora_B.' in name:
                noise=torch.randn(p.shape,generator=generator,dtype=torch.float32).to(p.device)
                norm=p.float().norm()
                if norm==0:raise ValueError('Randomized-B control requires a trained nonzero B')
                p.copy_((noise/noise.norm()*norm).to(p.dtype));changed.append(name)
    if not changed:raise ValueError('No LoRA B parameters found')
    return changed


def merge_residual(before,after,a,b,scale):
    import torch
    before=before.cpu().half();after=after.cpu().half()
    delta=(b.cpu().float()@a.cpu().float()*scale).half()
    expected=before+delta
    ulp=(torch.nextafter(expected,torch.full_like(expected,float('inf')))-expected).abs().float().clamp_min(torch.finfo(torch.float16).smallest_normal*torch.finfo(torch.float16).eps)
    residual=(after.float()-expected.float()).abs()/ulp
    return {'max_ulp':float(residual.max()),'max_abs':float((after.float()-expected.float()).abs().max()),
            'relative_delta':float((after.float()-before.float()).norm()/before.float().norm())}


def calibrate_tiny_merge():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(42)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).half(),
        LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name,p in model.named_parameters():
            if '.lora_B.' in name:p.normal_(0,.05)
    module='model.layers.0.self_attn.q_proj';layer=model.base_model.model.get_submodule(module)
    before=layer.base_layer.weight.detach().clone();a=layer.lora_A['default'].weight.detach().clone();b=layer.lora_B['default'].weight.detach().clone()
    cases={}
    for factor in [.5,1.,2.]:
        candidate=copy.deepcopy(model)
        for layer in candidate.modules():
            if isinstance(getattr(layer,'scaling',None),dict) and 'default' in layer.scaling:layer.scaling['default']*=factor
        merged=candidate.merge_and_unload(safe_merge=True)
        cases[str(factor)]=merge_residual(before,merged.get_submodule(module).weight.detach(),a,b,2.)
    tolerance=cases['1.0']['max_ulp']+1.
    result={'correct_max_ulp':cases['1.0']['max_ulp'],'tolerance_ulp':tolerance,'cases':cases,
            'wrong_scale_detected':all(cases[str(f)]['max_ulp']>tolerance for f in [.5,2.])}
    if not result['wrong_scale_detected']:raise ValueError('Calibration cannot distinguish wrong scale')
    return result
