"""Pre-run analytic storage and explicit memory allowances, not a measured peak."""
import argparse
import math
from pathlib import Path
from common import atomic_json,load_config,utc,sha256


def calculate(c):
    s=c['model_shape'];h=s['hidden_size'];kv=h*s['num_key_value_heads']//s['num_attention_heads'];f=s['intermediate_size'];L=s['num_hidden_layers'];V=s['vocab_size']
    shapes={'q_proj':(h,h),'k_proj':(kv,h),'v_proj':(kv,h),'o_proj':(h,h),'gate_proj':(f,h),'up_proj':(f,h),'down_proj':(h,f)}
    linears=L*sum(a*b for a,b in shapes.values());emb=V*h;other=L*(h+2*kv)+(2*L+1)*h
    blocks=sum(L*math.ceil(a*b/64) for a,b in shapes.values())
    dq_bytes=blocks+4*math.ceil(blocks/256)
    packed=linears//2;fp16_other=(emb+other)*2
    adapter=L*c['r']*sum(sum(shapes[name]) for name in c['targets'])
    b,seq=c['batch_size'],c['max_length'];scale=b*seq/512
    acts=[x*1e9*scale for x in c['activation_allowance_gb_at_bs1_512']]
    # prepare_model_for_kbit_training usually leaves trainable LoRA/nonquantized tensors FP32; census verifies this.
    weights=adapter*4;grads=adapter*4;adam=adapter*(2 if c['optimizer']=='paged_adamw_8bit' else 8)
    upcast=(emb+other)*2
    logits16=b*seq*V*2;logits32=logits16*2
    base=packed+fp16_other+dq_bytes;overhead=[x*1e9 for x in c['cuda_allowance_gb']]
    lower=base+upcast+weights+grads+adam+logits32+acts[0]+overhead[0]
    upper=base+upcast+weights+grads+adam+logits16+logits32+acts[1]+overhead[1]
    return {'parameters':linears+emb+other,'linear_parameters':linears,'embedding_parameters':emb,'bias_norm_parameters':other,
        'trainable_parameters':adapter,'packed_linear_bytes':packed,'embedding_fp16_bytes':emb*2,'bias_norm_fp16_bytes':other*2,
        'double_quant_stats_bytes':dq_bytes,'nf4_analytic_bytes':base,'nf4_floor_without_stats_bytes':packed+fp16_other,
        'expected_fp32_nonquantized_upcast_bytes':upcast,'adapter_weights_bytes':weights,'adapter_gradients_bytes':grads,
        'optimizer_state_bytes':adam,'logits_fp16_bytes':logits16,'logits_fp32_bytes':logits32,
        'checkpointed_hidden_anchor_bytes':L*b*seq*h*2,'activation_workspace_allowance_bytes':acts,
        'cuda_allocator_allowance_bytes':overhead,'training_device_envelope_bytes':[lower,upper],
        'merge_host_envelope_bytes':[(linears+emb+other)*4+1e9,(linears+emb+other)*4+4e9],
        'quantize_host_envelope_bytes':[(linears+emb+other)*2+base+1e9,(linears+emb+other)*2+base+3e9],
        'kind':'analytic plus uncalibrated allowances; allocator/RSS measurements must be compared after run',
        'batch_fallback_recommended':upper>13e9}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise SystemExit('Refusing to overwrite a pre-run budget')
    result=calculate(load_config(a.config));result.update(created_utc=utc(),config_sha256=sha256(a.config));atomic_json(a.out,result)
    print(result['training_device_envelope_bytes'])


if __name__=='__main__':main()
