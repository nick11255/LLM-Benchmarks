# engine.py
import torch

@torch.inference_mode()
def static_batch(prompts, model, tok, max_new_tokens=200):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    input_ids, attn = enc.input_ids, enc.attention_mask
    past = None
    B = input_ids.shape[0]
    generated = [[] for _ in range(B)]

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)   # [B, 1]
        for i in range(B):
            generated[i].append(next_tok[i].item())
        input_ids = next_tok
        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)

    return [tok.decode(seq, skip_special_tokens=True) for seq in generated]