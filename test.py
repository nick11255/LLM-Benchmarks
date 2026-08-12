# Cell 2
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-1.5B"        
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda")

ids = tok("Explain how a CPU cache works.", return_tensors="pt").to("cuda")
_ = model.generate(**ids, max_new_tokens=8)                      

t = time.perf_counter()
out = model.generate(**ids, max_new_tokens=200, do_sample=False)  
dt = time.perf_counter() - t

n = out.shape[1] - ids.input_ids.shape[1]
print(tok.decode(out[0], skip_special_tokens=True))
print(f"\n{n} tokens in {dt:.2f}s → {n/dt:.1f} tokens/sec")