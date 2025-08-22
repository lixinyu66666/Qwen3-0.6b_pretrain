import json, time, argparse, os, sys, torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.eval import distinct_n_ratio, generate_one

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--prompts", default=None, help="Path to a .txt; one prompt per line.")
    ap.add_argument("--out", default="results/sanity_outputs.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "Write a short poem about stars.",
            "Explain the theory of relativity in simple terms.",
            "What are the benefits of regular exercise?",
            "Describe a futuristic city.",
            "写一段关于秋天的散文。",
        ]
    
    with open(args.out, "w", encoding="utf-8") as fout:
        for p in prompts:
            start = time.time()
            text = generate_one(model, tok, p, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
            dur = time.time() - start
            toks = text.split()
            d1 = distinct_n_ratio(toks, 1)
            d2 = distinct_n_ratio(toks, 2)
            rec = {"prompt:": p, "output": text, "sec": round(dur, 4), "distinct_1": d1, "distinct_2": d2}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[OK] prompt len={len(p)} | time={dur:.2f}s | distinct1={d1:.3f} distinct2={d2:.3f}")

if __name__ == "__main__":
    main()