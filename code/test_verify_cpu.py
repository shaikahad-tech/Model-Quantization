import copy
import tempfile
import unittest
from pathlib import Path
from common import complete_stage,valid_stage
from model_checks import calibrate_tiny_merge,randomize_b


class VerifierChecks(unittest.TestCase):
    def test_resume_hash(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);p=run/'weights.txt';p.write_text('model')
            complete_stage(run,'train',[p],{})
            self.assertTrue(valid_stage(run,'train'));p.write_text('wrong');self.assertFalse(valid_stage(run,'train'))
    def test_merge_wrong_scale(self):
        result=calibrate_tiny_merge();self.assertTrue(result['wrong_scale_detected'])
        self.assertLessEqual(result['cases']['1.0']['max_ulp'],result['tolerance_ulp'])
    def test_noop_and_random_key_mapping(self):
        import torch
        from transformers import Qwen2Config,Qwen2ForCausalLM
        from peft import LoraConfig,get_peft_model
        torch.manual_seed(42)
        base=Qwen2ForCausalLM(Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).eval()
        ids=torch.tensor([[1,2,3]])
        with torch.no_grad():before=base(ids).logits
        model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
        with torch.no_grad():self.assertTrue(torch.allclose(before,model(ids).logits,atol=1e-5,rtol=1e-5))
        with self.assertRaises(ValueError):randomize_b(model)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if '.lora_B.' in name:p.normal_()
        old={n:p.detach().clone() for n,p in model.named_parameters()};changed=randomize_b(model)
        self.assertEqual(len(changed),4)
        for name,p in model.named_parameters():
            if name in changed:
                self.assertFalse(torch.equal(old[name],p));self.assertTrue(torch.allclose(old[name].norm(),p.norm(),rtol=1e-5))
            else:self.assertTrue(torch.equal(old[name],p))
