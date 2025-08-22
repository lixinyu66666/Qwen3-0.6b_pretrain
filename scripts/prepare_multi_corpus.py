"""
prepare_multi_corpus.py
-----------------------
Stream sampling from multiple Hugging Face sources, unified cleaning and counting, 
output Arrow shards, targeting cumulative N tokens.
Default for Chinese-English: RedPajama-v2-small(en) + Wiki(zh-cn/zh-tw/en) + StackOverflow(en).

Usage example (target 9B tokens, ~1e8 tokens per shard, see --mix for default ratios):
  python scripts/prepare_multi_corpus.py \
    --target_tokens 9e9 \
    --shard_tokens 1e8 \
    --out_dir data/processed \
    --tokenizer_dir tokenizer \
    --mix "wiki_en=0.55,rpjv2_small_en=0.2,wiki_zh=0.15,wiki_tw=0.05,stackoverflow_en=0.05"

Notes:
- Only counts tokens after tokenization with "your tokenizer", so token budget is accurate.
- Uses streaming datasets, no need to download complete datasets beforehand.
- StackOverflow: simple HTML removal; Wiki/RedPajama: preserve main text.
"""

import os, math, json, re, html, argparse, collections, sys
from typing import Dict, Tuple, Iterator, Optional
import pyarrow as pa, pyarrow.dataset as ds
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data.streaming_dataset import process_one_source, parse_mix, SOURCES

disable_caching()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", default="tokenizer")
    ap.add_argument("--total_tokens", type=float, default=9e9, help="Total tokens to collect (default 9B).")
    ap.add_argument("--shard_tokens", type=float, default=1e8, help="Tokens per Arrow shard (default 1e8).")
    ap.add_argument("--out_dir", default="data/processed", help="Output directory for Arrow shards.")
    ap.add_argument("--mix", default="wiki_en=0.55,wiki_zh=0.25,wiki_tw=0.1,stackoverflow_en=0.1")
    ap.add_argument("--min_chars", type=int, default=32)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    mix = parse_mix(args.mix)
    per_budget = {k: int(v * args.total_tokens) for k, v in mix.items()}
    for k, v in per_budget.items():
        print(f"{k}: {v/1e9:.2f}B tokens")
    
    for key, weight in mix.items():
        cfg, field, lang = SOURCES[key]
        process_one_source(
            key=key,
            cfg=cfg,
            field=field,
            lang=lang,
            min_chars=args.min_chars,
            target_tokens=per_budget[key],
            shard_tokens=args.shard_tokens,
            out_dir=os.path.join(args.out_dir, f"{key}_arrow"),
            tokenizer=tokenizer
        )
    print("All done.")

if __name__ == "__main__":
    main()