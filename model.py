
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelRunner:
    """loads model and tokeniser"""

    def __init__(self, name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate_single(self, prompt, max_new_tokens=200):
        """manual decode loop with batch size of 1"""
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        input_ids, attn = enc.input_ids, enc.attention_mask
        past = None
        out_tokens = []

        for _ in range(max_new_tokens):
            out = self.model(input_ids=input_ids, attention_mask=attn,
                             past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            out_tokens.append(next_tok.item())
            input_ids = next_tok
            attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)

        return self.tok.decode(out_tokens, skip_special_tokens=True)