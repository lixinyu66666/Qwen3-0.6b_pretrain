"""
train_tokenizer.py
------------------
Train a SentencePiece-BPE tokenizer from a text file.

Usage:
  python scripts/train_tokenizer.py \
      --input data/raw/tmp_corpus.txt \
      --out_dir tokenizer \
      --vocab_size 32000
"""

import argparse, os, sentencepiece as spm
from tokenizers import SentencePieceBPETokenizer, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the input text file.")
    ap.add_argument("--out_dir", default="tokenizer", help="Directory to save the tokenizer.")
    ap.add_argument("--vocab_size", type=int, default=32000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<pad>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        show_progress=True,
    )
    backend.train([args.input], trainer=trainer)

    tok_json = os.path.join(args.out_dir, "tokenizer.json")
    backend.save(tok_json)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=tok_json,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )

    hf_tok.model_max_length = 32768
    hf_tok.save_pretrained(args.out_dir)
    print(f"Tokenizer saved to {args.out_dir}")

if __name__ == "__main__":
    main()