from typing import Iterator, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os, sys
import pyarrow as pa, pyarrow.dataset as ds

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.clean import clean_stackoverflow, clean_generic

SOURCES = {
    "wiki_zh": (dict(path="wiki40b", name="zh-cn", streaming=True, split="train"), "text", "zh"),
    "wiki_tw": (dict(path="wiki40b", name="zh-tw", streaming=True, split="train"), "text", "zh"),
    "wiki_en": (dict(path="wiki40b", name="en", streaming=True, split="train"), "text", "en"),
    "stackoverflow_en": (dict(path="pacovaldez/stackoverflow-questions", streaming=False, split="train"), "body", "en"),
}

def get_text(sample: dict, field: str) -> str:
    v = sample.get(field, "")
    if not isinstance(v, str):
        v = str(v) if v is not None else ""
    return v

def iter_source(key: str, cfg: dict, field: str, lang: str, min_chars: int) -> Iterator[Tuple[str, str]]:
    ds = load_dataset(**cfg)
    if isinstance(ds, dict) and "train" in ds:
        ds = ds["train"]
    for sample in ds:
        text = get_text(sample, field)
        if not text:
            continue
        if key.startswith("stackoverflow"):
            text = clean_stackoverflow(text)
        else:
            text = clean_generic(text)
        
        if len(text) < min_chars:
            continue

        yield lang, text
    
def parse_mix(mix_str: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    parts = [p.strip() for p in mix_str.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        if k in SOURCES:
            try:
                out[k] = float(v.strip())
            except Exception:
                pass

    s = sum(out.values())
    if s <= 0:
        raise ValueError("Invalid --mix (sum <= 0 or keys not in SOURCES).")
    for k in list(out.keys()):
        out[k] = out[k] / s
    
    return out

def write_shard(rows, out_dir: str, shard_id: int):
    table = pa.Table.from_pylist(rows)
    p = os.path.join(out_dir, f"shard_{shard_id:04d}.arrow")
    ds.write_dataset(
        table,
        p,
        format="arrow",
        existing_data_behavior="overwrite_or_ignore",
    )

def process_one_source(
        key: str,
        cfg: dict,
        field: str,
        lang: str,
        tokenizer: AutoTokenizer,
        target_tokens: int,
        shard_tokens: int,
        out_dir: str,
        min_chars: int
):
    print(f"Processing {key}...")
    os.makedirs(out_dir, exist_ok=True)
    
    it = iter_source(key, cfg, field, lang, min_chars)
    shard_id = 0
    shard_tok = 0
    total_tok = 0
    buffer = []

    pbar = tqdm(total=target_tokens, unit="tok", desc=f"{key}")

    while total_tok < target_tokens:
        try:
            lang_, txt = next(it)  # 获取下一个文本样本
        except StopIteration:
            break

        n_tok = len(tokenizer.encode(txt, add_special_tokens=False))
        if n_tok <= 0:
            continue

        if total_tok + n_tok > int(target_tokens * 1.01):
            break

        buffer.append({"text": txt, "lang": lang_})
        shard_tok += n_tok
        total_tok += n_tok
        pbar.update(n_tok)
        pbar.set_postfix(total=f"{total_tok/1e9:.2f}B", shard=shard_id)

        if shard_tok >= shard_tokens:
            write_shard(buffer, out_dir, shard_id)
            buffer.clear()
            shard_tok = 0
            shard_id += 1
    
    if buffer:
        write_shard(buffer, out_dir, shard_id)
    
    pbar.close()
    print(f"[{key}] done: {total_tok/1e9:.3f} B tokens → {out_dir}")