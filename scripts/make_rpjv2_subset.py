"""make_rpjv2_subset.py
------------------------------------------------
Create a *subset* of RedPajama-V2 for pre-training.

• Streams data directly from HF Hub (no full download required).
• Counts tokens with your custom tokenizer so the budget is exact.
• Writes out Arrow shards (≈1 GiB each) ready for streaming.

Example
-------
```bash
python scripts/make_rpjv2_subset.py \
       --target_tokens 5e6 \
       --out_dir data/processed/rpjv2_debug
```
"""
import argparse, os, math, pyarrow as pa, pyarrow.dataset as ds
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer
from tqdm import tqdm
import json

disable_caching()

LANG_SHARE = {"en": 0.70, "de": 0.075, "fr": 0.075, "es": 0.075, "it": 0.075}
DEFAULT_PARTITION = "head_middle"
DEFAULT_NAME = "default"

def accept(sample):
    txt = sample.get("raw_content")
    if not txt or len(txt.strip()) < 32:
        return False, None, None
    
    lang = get_lang(sample)
    if lang not in LANG_SHARE:
        return False, None, None
    
    qs = sample.get("quality_signals")
    qs = json.loads(qs)
    ppl_raw = qs.get("ccnet_perplexity")
    ppl = ppl_raw[0][2]
    if ppl > 400:
        return False, None, None
    
    return True, lang, txt

def get_lang(sample):
    meta = sample.get("meta")
    if isinstance(meta, dict):
        lang = meta.get("language")
    else:
        meta = json.loads(meta)
        lang = meta.get("language")
    return lang

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=float, default=3e9, help="total tokens to collect (default 3B)")
    ap.add_argument("--shard_tokens", type=float, default=1e8, help="write one Arrow shard per N tokens (default 1e8)")
    ap.add_argument("--out_dir", default="data/processed/rpjv2_3B_arrow")
    ap.add_argument("--partition", default=DEFAULT_PARTITION)
    ap.add_argument("--name", default=DEFAULT_NAME)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    if (tokenizer.unk_token is None) or (tokenizer.unk_token not in tokenizer.get_vocab()):
        tokenizer.add_special_tokens({'unk_token': '<unk>'})
        tokenizer.save_pretrained("tokenizer")
    
    stream_ds = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        streaming=True,
        split="train",
        partition=args.partition,
        name=args.name,
        trust_remote_code=True,
    )

    token_total, shard_token_cnt = 0, 0
    lang_counter = {k: 0 for k in LANG_SHARE}
    shard_id, buffer = 0, []

    pbar = tqdm(stream_ds, desc="collect", unit="docs")
    for sample in pbar:
        keep, lang, text = accept(sample)
        if not keep:
            continue

        # if token_total and lang_counter[lang] / token_total > LANG_SHARE[lang] + 0.01:
        #     continue
        
        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        token_total += n_tok
        shard_token_cnt += n_tok
        lang_counter[lang] += n_tok
        buffer.append({"text": text, "lang": lang})

        pbar.set_postfix(
            tokens=f"{token_total/1e6:.1f}M", 
            shard=shard_id,
            lang=lang)
        
        if shard_token_cnt >= args.shard_tokens:
            table = pa.Table.from_pylist(buffer)
            ds.write_dataset(
                table,
                f"{args.out_dir}/shard_{shard_id:04d}.arrow",
                format="arrow",
                existing_data_behavior="overwrite_or_ignore",
            )
            shard_id += 1
            buffer, shard_token_cnt = [], 0

        if token_total >= args.target_tokens:
            break

    if buffer:
        table = pa.Table.from_pylist(buffer)
        ds.write_dataset(
            table,
            f"{args.out_dir}/shard_{shard_id:04d}.arrow",
            format="arrow",
            existing_data_behavior="overwrite_or_ignore",
        )
    
    print(f"Done. {token_total/1e6:.2f} M tokens saved in {args.out_dir}")

def debug(n=5):
    ds = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        name="default",
        split="train",
        streaming=True,
        partition="head_middle",
        trust_remote_code=True,
    )

    for i, sample in enumerate(ds):
        print("\n" + "=" * 50)
        print(f"SAMPLE #{i}")
        print("type:", type(sample).__name__)
        print("keys:", list(sample.keys()))

        for k, v in sample.items():
            if k == "raw_content":
                txt = v if isinstance(v, str) else str(v)
                pv = txt.replace("\n", " ")
                print(f"{k} (preview):", pv[:200] + (" ...[truncated]" if len(pv) > 200 else ""))

            elif k == "meta":
                if isinstance(v, dict):
                    print(f"{k} (dict):")
                    for mk, mv in v.items():
                        print(f"    {mk}: {mv}")
                else:
                    try:
                        meta_dict = json.loads(v)
                        print(f"{k} (parsed from str):")
                        for mk, mv in meta_dict.items():
                            print(f"    {mk}: {mv}")
                    except Exception:
                        print(f"{k} (raw):", v)

            elif k == "quality_signals":
                if isinstance(v, dict):
                    print(f"{k} (dict):")
                    for qk, qv in v.items():
                        print(f"    {qk}: {qv}")
                else:
                    try:
                        qs_dict = json.loads(v)
                        print(f"{k} (parsed from str):")
                        for qk, qv in qs_dict.items():
                            print(f"    {qk}: {qv}")
                    except Exception:
                        print(f"{k} (raw):", v)

            else:
                print(f"{k}:", v)

        if i + 1 >= n:
            break

if __name__ == "__main__":
    main()
    # debug()