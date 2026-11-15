"""Unexecuted GPU checks; no shipping verdict. Original eight-row script is unchanged."""
import gc
import json
import math
from pathlib import Path

from experiment_data import BASE, REVISION, encode, read_data


def verify(run, tok):
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    _, rows = read_data(root / 'data/sampled_dataset.zip')
    features = [encode(tok, row) for row in rows]
    tasks = [json.loads(x) for x in (root / 'data/ship_tasks.jsonl').read_text().splitlines()]
    result = {'executed_rows_per_model': 100, 'base_revision': REVISION, 'artifacts': {},
              'controls': 'No-op LoRA and seeded randomized B with trained A and matched B norms',
              'safety': 'NOT EVALUATED: original safety cases/rubric unavailable',
              'shuffled_label_control': 'NOT RUN', 'release_decision': "DON'T SHIP"}
    saved = {}
    with safe_open(run / 'adapter/adapter_model.safetensors', framework='pt', device='cpu') as f:
        saved = {k: f.get_tensor(k).float() for k in f.keys()}
    b_norms = {k: v.norm().item() for k, v in saved.items() if k.endswith('.lora_B.weight')}
    result['adapter_B_norms'] = b_norms
    result['adapter_B_nonzero'] = sum(math.isfinite(n) and n > 0 for n in b_norms.values())
    modules = [k.removeprefix('base_model.model.').removesuffix('.lora_A.weight')
               for k in saved if k.endswith('.lora_A.weight')]
    assert len(modules) == 56 and all(m.endswith(('q_proj', 'v_proj')) for m in modules)
    cfg = json.loads((run / 'adapter/adapter_config.json').read_text())
    assert cfg['r'] == 16 and cfg['lora_alpha'] == 32 and not cfg.get('use_rslora') and not cfg.get('use_dora')
    assert not cfg.get('rank_pattern') and not cfg.get('alpha_pattern') and not cfg.get('modules_to_save')
    assert cfg.get('bias') == 'none'
    original_weights, fixed_logits = {}, {}
    question = 'Why is a bicycle helmet useful?'
    fixed_ids = tok.apply_chat_template([{'role': 'user', 'content': question}], tokenize=True,
                                        add_generation_prompt=True, return_tensors='pt').cuda()
    for name in ['base', 'noop', 'randomized_B', 'adapter', 'merged', '4bit']:
        path = BASE if name in ['base', 'noop', 'randomized_B', 'adapter'] else run / ('quantized' if name == '4bit' else name)
        kwargs = {'revision': REVISION} if path == BASE else {}
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map={'': 0}, **kwargs)
        artifact_tok = AutoTokenizer.from_pretrained(path, **kwargs)
        assert artifact_tok.get_vocab() == tok.get_vocab() and artifact_tok.chat_template == tok.chat_template
        assert artifact_tok.eos_token_id == tok.eos_token_id and artifact_tok.pad_token_id == tok.pad_token_id
        if name == 'base':
            original_weights = {m: model.get_submodule(m).weight.detach().float().cpu() for m in modules}
        if name == 'adapter':
            model = PeftModel.from_pretrained(model, run / 'adapter')
        elif name in ['noop', 'randomized_B']:
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'],
                                                   bias='none', task_type='CAUSAL_LM'))
            if name == 'randomized_B':
                torch.manual_seed(20261111)
                with torch.no_grad():
                    for key, parameter in model.named_parameters():
                        source = saved.get(key.replace('.default.', '.'))
                        if source is not None:
                            if 'lora_B' in key:
                                noise = torch.randn_like(parameter)
                                parameter.copy_(noise * (source.norm().to(noise.device) / noise.norm()))
                            else:
                                parameter.copy_(source.to(parameter.device))
        if name == 'merged':
            checks = []
            for m in modules:
                a = saved['base_model.model.' + m + '.lora_A.weight']
                b = saved['base_model.model.' + m + '.lora_B.weight']
                before = original_weights[m]
                after = model.get_submodule(m).weight.detach().float().cpu()
                expected = (before + (cfg['lora_alpha'] / cfg['r']) * (b @ a)).half().float()
                checks.append({'module': m, 'relative_delta': ((after-before).norm()/before.norm()).item(),
                    'max_merge_residual': (after-expected).abs().max().item(),
                    'within_fp16_tolerance': torch.allclose(after, expected, atol=1e-4, rtol=1e-3)})
            result['merge_checks'] = checks
        if name == '4bit':
            converted = {n: str(m.compute_dtype) for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)}
            expected_names = {f'model.layers.{i}.{part}.{proj}' for i in range(28)
                for part, projections in [('self_attn', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                                           ('mlp', ['gate_proj', 'up_proj', 'down_proj'])] for proj in projections}
            result['quantized_modules'] = converted
            result['quantized_config'] = model.config.quantization_config.to_dict() if hasattr(model.config.quantization_config, 'to_dict') else model.config.quantization_config
            result['quantized_coverage_pass'] = set(converted) == expected_names
            result['quantized_dtype_pass'] = all(v == 'torch.float16' for v in converted.values())
            quant = result['quantized_config']
            result['quantized_format_pass'] = bool(quant.get('load_in_4bit') and
                quant.get('bnb_4bit_quant_type') == 'nf4' and quant.get('bnb_4bit_use_double_quant'))
        model.eval()
        cases, task_cases = [], []
        with torch.inference_mode():
            fixed_logits[name] = model(fixed_ids).logits[0, -1].float().cpu()
            for i, feature in enumerate(features):
                ids = torch.tensor([feature['input_ids']], device='cuda')
                start = feature['labels'].count(-100)
                logits = model(ids).logits[:, start-1:-1].float()
                targets = ids[:, start:]
                loss_sum = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='sum').item()
                cases.append({'index': i, 'target_tokens': targets.numel(), 'nll_sum': loss_sum,
                              'nll_per_token': loss_sum / targets.numel()})
                del logits, targets, ids
            for i, task in enumerate(tasks):
                ids = tok.apply_chat_template([{'role': 'user', 'content': task['prompt']}], tokenize=True,
                    add_generation_prompt=True, return_tensors='pt').cuda()
                answer_ids = model.generate(ids, do_sample=False, max_new_tokens=64, pad_token_id=tok.pad_token_id)
                answer = tok.decode(answer_ids[0, ids.shape[1]:], skip_special_tokens=True)
                task_cases.append({'index': i, 'prompt': task['prompt'], 'expected': task['expected'],
                    'answer': answer, 'contains_case_sensitive': task['expected'] in answer})
        count = sum(c['target_tokens'] for c in cases)
        result['artifacts'][name] = {'rows': cases, 'target_tokens': count,
            'nll_per_token': sum(c['nll_sum'] for c in cases)/count, 'task_cases': task_cases,
            'task_passes': sum(c['contains_case_sensitive'] for c in task_cases), 'task_denominator': len(task_cases)}
        (run / 'verification_extended.partial.json').write_text(json.dumps(result, indent=2))
        print(name, 'NLL', result['artifacts'][name]['nll_per_token'], 'rows', len(cases), 'tokens', count, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result['fixed_prompt'] = question
    result['fixed_logits'] = {name: {'relative_l2_vs_base': ((value-fixed_logits['base']).norm()/fixed_logits['base'].norm()).item(),
        'max_abs_vs_base': (value-fixed_logits['base']).abs().max().item()} for name, value in fixed_logits.items()}
    result['adapter_merged_logits_max_abs'] = (fixed_logits['adapter']-fixed_logits['merged']).abs().max().item()
    result['noop_logits_match'] = torch.allclose(fixed_logits['base'], fixed_logits['noop'], atol=1e-5, rtol=1e-5)
    result['learned_nll_beats_controls'] = all(result['artifacts']['adapter']['nll_per_token'] < result['artifacts'][n]['nll_per_token'] for n in ['base', 'noop', 'randomized_B'])
    result['finite_nll'] = all(math.isfinite(c['nll_per_token']) for a in result['artifacts'].values() for c in a['rows'])
    result['training_signal_pass'] = (result['adapter_B_nonzero'] == 56 and
        result['learned_nll_beats_controls'] and result['finite_nll'] and
        all(c['relative_delta'] > 1e-6 for c in result['merge_checks']))
    result['limitations'] = 'One random control, one logit prompt, no shuffled-label adapter, no safety scoring or uncertainty estimate; NLL gains do not prove useful learning.'
    torch.save(fixed_logits, run / 'fixed_prompt_logits.pt')
    (run / 'verification_extended.json').write_text(json.dumps(result, indent=2))
    assert result['noop_logits_match'] and result['quantized_coverage_pass'] and result['quantized_dtype_pass'] and result['quantized_format_pass']
    assert all(c['within_fp16_tolerance'] for c in result['merge_checks'])
    assert result['training_signal_pass'], 'No reliable signal against the implemented controls; inspect saved results'
