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
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the input text file.")
    ap.add_argument("--out_dir", default="tokenizer", help="Directory to save the tokenizer.")
    ap.add_argument("--vocab_size", type=int, default=32000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_prefix = os.path.join(args.out_dir, "qwen3")

    # spm.SentencePieceTrainer.train(
    #     input=args.input,
    #     model_prefix=model_prefix,
    #     vocab_size=args.vocab_size,
    #     model_type="bpe",
    #     character_coverage=0.9995,
    #     user_defined_symbols=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    # )

    tok = SentencePieceBPETokenizer()
    tok.train(
        files=[args.input],
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    )

    tok_json_path = os.path.join(args.out_dir, "tokenizer.json")
    tok.save(tok_json_path)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=tok_json_path,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|endoftext|>",
        add_bos_token=False,
        add_eos_token=False,
    )

    hf_tok.model_max_length = 32768
    hf_tok.save_pretrained(args.out_dir)
    print(f"Tokenizer saved to {args.out_dir}")

if __name__ == "__main__":
    main()