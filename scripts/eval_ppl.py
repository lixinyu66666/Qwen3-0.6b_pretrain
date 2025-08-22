import os, argparse, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.eval import ppl_on_texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to the model checkpoint.")
    ap.add_argument("--tokenizer", default="tokenizer", help="Path to the tokenizer.")
    ap.add_argument("--dataset", default=None, help="Dataset name for evaluation.")
    ap.add_argument("--subset", default=None, help="Subset of the dataset to evaluate on.")
    ap.add_argument("--split", default="test", help="Dataset split to evaluate on.")
    ap.add_argument("--max_samples", type=int, default=2000, help="Max samples for HF evaluation.")
    ap.add_argument("--block_size", type=int, default=2048, help="Block size for evaluation.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    texts = []
    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    texts = [ex.get("text", "") for ex in dataset]

    ppl = ppl_on_texts(model, tok, texts, block_size=args.block_size)
    print(f"Perplexity: {ppl}")

if __name__ == "__main__":
    main()