"""New HF workflow. GPU execution pending; never overwrites historical evidence."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

from experiment_data import BASE, REVISION, TEMPLATE_HASH, encode, read_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'merge', 'quantize', 'verify'])
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run == ROOT or ROOT / 'evidence' == run or ROOT / 'code' == run:
        raise ValueError('Use a separate run directory')
    run.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError('Requires Linux NVIDIA CUDA; this workflow does not support Mac MPS')
    set_seed(20261111)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    if hashlib.sha256(tok.chat_template.encode()).hexdigest() != TEMPLATE_HASH:
        raise ValueError('Pinned template mismatch')
    qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    adapter, merged, quantized = (run / n for n in ['adapter', 'merged', 'quantized'])
    outputs = {'train': adapter, 'merge': merged, 'quantize': quantized, 'verify': run / 'verification_extended.json'}
    if outputs[args.stage].exists():
        raise FileExistsError('Refusing to overwrite ' + str(outputs[args.stage]))
    torch.cuda.reset_peak_memory_stats()
    if args.stage == 'train':
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments, default_data_collator
        train, heldout = read_data(ROOT / 'data/sampled_dataset.zip')
        features = [encode(tok, r) for r in train]
        heldout_features = [encode(tok, r) for r in heldout]
        (run / 'preflight.json').write_text(json.dumps({
            'base_revision': REVISION, 'template_sha256': TEMPLATE_HASH,
            'rows_scanned': len(features) + len(heldout_features),
            'train': 1260, 'validation': 140, 'heldout': 100,
            'padding': 'none; batch size 1', 'label_policy': 'prompt=-100; assistant incl EOS supervised',
            'train_target_tokens': [sum(x != -100 for x in r['labels']) for r in features],
        }, indent=2))
        model = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION,
            quantization_config=qconfig, torch_dtype=torch.float16, device_map={'': 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'],
            lora_dropout=0.0, bias='none', task_type='CAUSAL_LM'))
        model.config.use_cache = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 2179072, trainable
        config = TrainingArguments(output_dir=str(run / 'checkpoints'), num_train_epochs=1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=2e-4, fp16=True, gradient_checkpointing=True, logging_steps=10,
            eval_strategy='epoch', save_strategy='no', report_to='none', seed=20261111,
            data_seed=20261111, optim='adamw_torch', weight_decay=0.0)
        trainer = Trainer(model=model, args=config, train_dataset=Dataset.from_list(features[:1260]),
            eval_dataset=Dataset.from_list(features[1260:]), data_collator=default_data_collator)
        trainer.train()
        trainer.save_model(str(adapter))
        tok.save_pretrained(adapter)
    elif args.stage == 'merge':
        base = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION,
            torch_dtype=torch.float16, device_map={'': 'cpu'})
        model = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
        model.save_pretrained(merged, safe_serialization=True)
        tok.save_pretrained(merged)
    elif args.stage == 'quantize':
        model = AutoModelForCausalLM.from_pretrained(merged, quantization_config=qconfig,
            torch_dtype=torch.float16, device_map={'': 0})
        model.save_pretrained(quantized, safe_serialization=True)
        tok.save_pretrained(quantized)
    else:
        from verify_extended import verify
        verify(run, tok)
    torch.cuda.synchronize()
    print(json.dumps({'stage': args.stage, 'max_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'max_reserved_bytes': torch.cuda.max_memory_reserved()}), flush=True)
    gc.collect()


if __name__ == '__main__':
    main()
