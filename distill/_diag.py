import os, time
# Force offline: weights are cached, so don't let network checks stall the load.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"

print("cuda:", torch.cuda.is_available())
free, total = torch.cuda.mem_get_info()
print(f"vram free/total GB: {free/1e9:.2f}/{total/1e9:.2f}")

t0 = time.time()
tok = AutoTokenizer.from_pretrained(BASE)
print(f"tokenizer load: {time.time()-t0:.1f}s")

t0 = time.time()
# Explicit single-GPU placement — NO device_map='auto' (which can offload to CPU).
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
model = model.to("cuda")
model.eval()
print(f"model load+to(cuda): {time.time()-t0:.1f}s")

devs = {str(p.device) for p in model.parameters()}
print("param devices:", devs)
after_free, _ = torch.cuda.mem_get_info()
print(f"vram used by model GB: {(free-after_free)/1e9:.2f}")

# One real generation, timed.
msgs = [{"role": "user", "content": "Analyze the zone_temperature trend in zone_a over the past week."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to("cuda")
# warmup
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=8, do_sample=False,
                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
torch.cuda.synchronize()
n_new = out.shape[1] - inputs["input_ids"].shape[1]
dt = time.time() - t0
print(f"generate {n_new} tokens: {dt:.2f}s  ({n_new/dt:.1f} tok/s)")
print("DIAG DONE")
