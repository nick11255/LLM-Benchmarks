
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
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)   
        for i in range(B):
            generated[i].append(next_tok[i].item())
        input_ids = next_tok
        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)

    return [tok.decode(seq, skip_special_tokens=True) for seq in generated]
@torch.inference_mode()
def dynamic_batch(prompts, model, tok, max_new_tokens=200):
    eos_ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        eos_ids.add(im_end)
    prompts = [                               
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    input_ids, attn = enc.input_ids, enc.attention_mask
    past = None

    B = input_ids.shape[0]
    outputs = [[] for _ in range(B)]   
    active = list(range(B))            

    for step in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks = next_tok.squeeze(1).tolist()

        keep = []
        for row, orig in enumerate(active):
            outputs[orig].append(toks[row])
            if toks[row] not in eos_ids:
                keep.append(row)

        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
        input_ids = next_tok

        if len(keep) == 0:
            break
        if len(keep) < len(active):
            print(f"step {step}: {len(active) - len(keep)} finished, {len(keep)} still going")
            keep_idx = torch.tensor(keep, device=model.device)
            past.reorder_cache(keep_idx)    
            input_ids = input_ids[keep_idx]
            attn = attn[keep_idx]
            active = [active[r] for r in keep]

    return [tok.decode(o, skip_special_tokens=True) for o in outputs]