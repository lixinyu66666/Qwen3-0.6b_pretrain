"""
prepare_tmp_corpus.py
---------------------
Grab a small mixed-language corpus (en + zh) for tokenizer training.

Usage:
  python scripts/prepare_tmp_corpus.py \
      --out data/raw/tmp_corpus.txt \
      --max_mb 200          # stop when file ~200 MB
      --max_lines 200000    # (alternative) stop at N lines
"""

import argparse, os, io, gzip, random, itertools, warnings, sys
from datasets import load_dataset, logging as ds_logging
from tqdm import tqdm

ds_logging.set_verbosity_error()
warnings.filterwarnings("ignore")   

SOURCES = {
    "rpjv2_small_en": (dict(path="vilm/RedPajama-v2-small", streaming=True, split="train"), "text"),
    "wiki_zh": (dict(path="wiki40b", name="zh-cn", streaming=True, split="train"), "text"),
    "wiki_tw": (dict(path="wiki40b", name="zh-tw", streaming=True, split="train"), "text"),
    "wiki_en": (dict(path="wiki40b", name="en", streaming=True, split="train"), "text"),
    "stackoverflow_en": (dict(path="pacovaldez/stackoverflow-questions", streaming=False, split="train[:1%]"), "body"),
}

def iter_sources(shuffle=False):
    iters = {k: iter(load_dataset(**cfg[0])) for k, cfg in SOURCES.items()}
    keys = list(iters)
    while keys:
        for k in list(keys):
            try:
                sample = next(iters[k])
                text = str(sample[SOURCES[k][1]]).strip()
                if text:
                    yield text.replace("\n", " ")
            
            except StopIteration:
                keys.remove(k)
    
    if shuffle:
        random.shuffle(keys)

def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    
    return f"{num_bytes:.1f}TB"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/tmp_corpus.txt")
    ap.add_argument("--max_mb", type=int, default=100, help="target file size (MB) (exclusive with --max_lines)")
    ap.add_argument("--max_lines", type=int, default=None, help="stop after this many lines (if given)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    max_bytes = None if args.max_lines else args.max_mb * 1024 * 1024
    bytes_written = 0
    lines_written = 0

    with io.open(args.out, "w", encoding="utf-8") as fout:
        for txt in tqdm(iter_sources(), desc="collect"):
            fout.write(txt + "\n")
            bytes_written += len(txt.encode("utf-8")) + 1
            lines_written += 1

            if max_bytes and bytes_written >= max_bytes:
                break
            if args.max_lines and lines_written >= args.max_lines:
                break
    
    print(f"Written {lines_written} lines, {human_size(bytes_written)} to {args.out}")


if __name__ == "__main__":
    main()