prompts = [
   
]
#continous
@torch.inference_mode()
def generate_batch(prompts, max_new_tokens=200):
    tok.padding_side = "left"
  
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    input_ids = enc.input_ids          # now [batch, seq_len]
    attn = enc.attention_mask          # 0 = padding, 1 = real token
    past = None
    batch = input_ids.shape[0]


    generated = [[] for _ in range(batch)]
    for step in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attn,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values

        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)  # [batch, 1] — identical to your loop


        for i in range(batch):
            generated[i].append(next_tok[i].item())
        input_ids = next_tok
        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)

  
    return  [tok.decode(seq, skip_special_tokens=True) for seq in generated]

t = time.perf_counter()
outputs = generate_batch(prompts)
dt = time.perf_counter() - t
for o in outputs:
    print(o[:80], "...")

total = len(prompts) * 200
print(f"\n{total} tokens across {len(prompts)} sequences in {dt:.2f}s → {total/dt:.1f} tokens/sec")