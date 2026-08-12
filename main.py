import torch, time
#manual generation
@torch.inference_mode()
def generate_manual(prompt, max_new_tokens=200):
    enc = tok(prompt, return_tensors="pt").to("cuda")
    input_ids = enc.input_ids       
    attn = enc.attention_mask
    past = None
    generated = []

    for step in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values    

     
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)  

        generated.append(next_tok.item())


        input_ids = next_tok             
        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)

    return tok.decode(generated, skip_special_tokens=True)

t = time.perf_counter()
text = generate_manual("Explain how a CPU cache works.")
dt = time.perf_counter() - t
print(text)
print(f"\n200 tokens in {dt:.2f}s → {200/dt:.1f} tokens/sec")