import os, sys, time, torch, psutil, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.eval import count_new_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--prompt", default="Hello, world!")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    enc = tok(args.prompt, return_tensors="pt")
    enc.pop("token_type_ids", None)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        repetition_penalty=1.1,
    )
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model.generate(**enc, **gen_kwargs)
    
    latencies, tps = [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for _ in range(args.runs):
        start = time.time()
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        dur = time.time() - start
        new_tokens = count_new_tokens(enc["input_ids"], out)
        latencies.append(dur)
        tps.append(new_tokens / dur if dur > 0 else 0)
    
    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"avg_latency_s={sum(latencies)/len(latencies):.3f}  avg_tokens_per_s={sum(tps)/len(tps):.1f}")
    if peak_mem is not None:
        print(f"peak_cuda_mem_GB={peak_mem:.2f}")

    
if __name__ == "__main__":
    main()