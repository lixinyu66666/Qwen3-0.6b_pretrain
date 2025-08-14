import torch
from torch.utils.data import DataLoader 
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.packed_ds import NpzPackedDataset, simple_collate

def main():
    ds = NpzPackedDataset("data/packed/rpjv2_s2048", shuffle_files=True, seed=123)
    dl = DataLoader(
        ds, 
        batch_size=4,
        num_workers=8,
        pin_memory=True,
        collate_fn=simple_collate,
        prefetch_factor=2,  
        persistent_workers=True,
    )

    if hasattr(ds, "set_epoch"):
        ds.set_epoch(0)
    
    it = iter(dl)
    batch = next(it)
    print({k: v.shape for k, v in batch.items()})
    print("dtype:", batch["input_ids"].dtype, batch["attention_mask"].dtype)
    print("first row head:", batch["input_ids"][0, :32])

if __name__ == "__main__":
    main()