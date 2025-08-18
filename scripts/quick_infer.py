"""
Quick text generation to sanity-check a trained CausalLM.

Usage:
  python scripts/quick_infer.py \
    --ckpt checkpoints/qwen3_0p6b/step_00030000 \
    --tokenizer tokenizer \
    --prompt "Write a short poem about stars." \
    --max_new_tokens 128
"""

import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to the model checkpoint.")
    ap.add_argument("--tokenizer", default="tokenizer", help="Path to the tokenizer.")
    ap.add_argument("--prompt", default="Hello!", help="Input prompt for text generation.")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    ap.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling threshold.")
    ap.add_argument("--top_k", type=int, default=0, help="Top-k sampling threshold.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k > 0 else None,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id
        )
    print(tok.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()