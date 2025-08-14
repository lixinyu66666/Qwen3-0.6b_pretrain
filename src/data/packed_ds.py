import os, glob, numpy as np, torch
from torch.utils.data import IterableDataset

class NpzPackedDataset(IterableDataset):
    def __init__(self, root, shuffle_files=False, seed=42):
        self.root = root
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.files = sorted(glob.glob(os.path.join(root, "pack_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found in {root}")
    
    def set_epoch(self, epoch):
        self._epoch = epoch
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        files = self.files

        if worker is not None:
            files = files[worker.id::worker.num_workers]
        
        if getattr(self, "_epoch", None) is not None and self.shuffle_files:
            rng = np.random.RandomState(self.seed + int(self._epoch))
            idx = rng.permutation(len(files))
            files = [files[i] for i in idx]
        
        for f in files:
            dat = np.load(f)
            X = dat["input_ids"]          # [N, L], int32
            M = dat["attention_mask"]     # [N, L], int16/uint8
            for i in range(X.shape[0]):
                yield {
                    "input_ids": torch.as_tensor(X[i], dtype=torch.long),
                    "attention_mask": torch.as_tensor(M[i], dtype=torch.long),
                }

def simple_collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    
    return out