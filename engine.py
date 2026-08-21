
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

from collections import deque

@torch.inference_mode()
def continuous_batch(prompts, model, tok, max_batch=3, max_new_tokens=200):
    eos_ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(im_end)
    pad_id = tok.pad_token_id

   
    queue = deque()
    for i, p in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids[0].tolist()
        queue.append({"orig": i, "tokens": ids})

    outputs = [[] for _ in prompts]
    remaining = [max_new_tokens for _ in prompts]

    active = []         
    past = attn = next_tok = None
    need_prefill = False
    step = 0

    def admit():
        nonlocal need_prefill
        while queue and len(active) < max_batch:
            seq = queue.popleft()
            active.append(seq)
            need_prefill = True   
            print(f"  admit seq {seq['orig']}  (batch now {len(active)}, queue {len(queue)})")

    admit()                    

    while active:
        step += 1

        if need_prefill:
           
            seqs = [a["tokens"] for a in active]
            L = max(len(s) for s in seqs)
            input_ids = torch.tensor(
                [[pad_id] * (L - len(s)) + s for s in seqs], device=model.device)
            attn = torch.tensor(
                [[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=model.device)
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
            need_prefill = False
        else:
           
            out = model(input_ids=next_tok, attention_mask=attn,
                        past_key_values=past, use_cache=True)

        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
        toks = next_tok.squeeze(1).tolist()

        keep = []
        for row, a in enumerate(active):
            t = toks[row]
            a["tokens"].append(t)
            outputs[a["orig"]].append(t)
            remaining[a["orig"]] -= 1
            if (t in eos_ids) or (remaining[a["orig"]] <= 0):
                print(f"step {step}: seq {a['orig']} finished")
            else:
                keep.append(row)

        if len(keep) < len(active):         
            if keep:
                keep_idx = torch.tensor(keep, device=model.device)
                past.reorder_cache(keep_idx)
                next_tok = next_tok[keep_idx]
                attn = attn[keep_idx]
            active = [active[r] for r in keep]
            admit()                      

    return [tok.decode(o, skip_special_tokens=True) for o in outputs]