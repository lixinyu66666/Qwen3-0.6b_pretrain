"""
make_packed_dataset.py
----------------------
Arrow -> token ids -> fixed-length packed arrays (NPZ shards).

Usage:
  python scripts/make_packed_dataset.py \
    --arrow_dir data/processed/rpjv2_3B_arrow \
    --out_dir   data/packed/rpjv2_s2048 \
    --seq_len   2048 \
    --shard_tokens 50000000
"""
import os, argparse, numpy as np, pyarrow as pa, pyarrow.dataset as ds
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow_dir", required=True, help="Path to the input Arrow dataset directory.")
    ap.add_argument("--out_dir", default="data/packed/rpjv2_s2048", help="Directory to save the packed NPZ files.")
    ap.add_argument("--seq_len", type=int, default=2048, help="Sequence length for each packed example.")
    ap.add_argument("--shard_tokens", type=int, default=50000000, help="Number of tokens per NPZ shard.")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("tokenizer")

    dataset = ds.dataset(args.arrow_dir, format="arrow")
    scanner = ds.Scanner.from_dataset(dataset, columns=["text"], batch_size=8192)

    seq_len = args.seq_len
    shard_tokens = args.shard_tokens
    shard_id = 0
    token_since_last_flush = 0

    pool = []
    buf  = []

    def flush():
        nonlocal buf, shard_id, token_since_last_flush
        if not buf:
            return
        arr = np.stack(buf).astype(np.int32)                # [N, seq_len]
        attn = (arr != 0).astype(np.int16)
        out_path = os.path.join(args.out_dir, f"pack_{shard_id:04d}.npz")
        np.savez_compressed(out_path, input_ids=arr, attention_mask=attn)
        shard_id += 1
        buf.clear()
        token_since_last_flush = 0
    
    seen_rows = 0
    pbar = tqdm(scanner.to_batches(), desc="packing", unit="batch")
    for batch in pbar:
        rows = batch.to_pydict()["text"]
        for text in rows:
            if text is None:
                continue
            s = text if isinstance(text, str) else str(text)
            ids = tok.encode(s, add_special_tokens=False)
            eos_id = tok.eos_token_id
            if eos_id is not None:
                ids.append(eos_id)
            pool.extend(ids)
            seen_rows += 1
            
            while len(pool) >= seq_len:
                buf.append(np.array(pool[:seq_len], dtype=np.int32))
                del pool[:seq_len]
                token_since_last_flush += seq_len

                if token_since_last_flush >= shard_tokens:
                    flush()
            
            if args.max_rows > 0 and seen_rows >= args.max_rows:
                break
        
        if args.max_rows > 0 and seen_rows >= args.max_rows:
            break
        pbar.set_postfix(shard=shard_id, pool=len(pool), packed=len(buf))
    
    if len(pool) >= seq_len // 2:
        pad_len = seq_len - len(pool)
        buf.append(np.array(pool + [0] * pad_len, dtype=np.int32))

    flush()
    print("Done.")

if __name__ == "__main__":
    main()