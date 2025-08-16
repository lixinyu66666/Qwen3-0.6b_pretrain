import os, glob, numpy as np, torch
from torch.utils.data import IterableDataset
from typing import Union, List, Sequence

class NpzPackedDataset(IterableDataset):
    def __init__(
        self,
        root: Union[str, Sequence[str]],
        shuffle_files: bool = False,
        shuffle_rows: bool = False,
        seed: int = 42,
        pattern: str = "pack_*.npz",
    ):
        self.roots: List[str] = [root] if isinstance(root, str) else list(root)
        self.shuffle_files = shuffle_files
        self.shuffle_rows = shuffle_rows
        self.seed = int(seed)
        self.pattern = pattern

        files: List[str] = []
        for r in self.roots:
            files.extend(sorted(glob.glob(os.path.join(r, pattern))))
        self.files = files
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found in {[os.path.abspath(r) for r in self.roots]} (pattern={pattern})")

        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
    
    def _files_for_worker(self):
        worker = torch.utils.data.get_worker_info()
        files = self.files

        if self.shuffle_files:
            rng = np.random.RandomState(self.seed + self._epoch)
            idx = rng.permutation(len(files))
            files = [files[i] for i in idx]

        if worker is not None:
            files = files[worker.id::worker.num_workers]

        return files

    def __iter__(self):
        files = self._files_for_worker()

        for f in files:
            dat = np.load(f, mmap_mode="r", allow_pickle=False)
            X = dat["input_ids"]          # [N, L], int32
            M = dat["attention_mask"]     # [N, L], int16/uint8/bool
            n = X.shape[0]

            if self.shuffle_rows and n > 1:
                rng = np.random.RandomState(self.seed + self._epoch + (hash(f) % 10_000_000))
                order = rng.permutation(n)
            else:
                order = range(n)

            for i in order:
                yield {
                    "input_ids": torch.from_numpy(X[i].astype(np.int64, copy=False)),
                    "attention_mask": torch.from_numpy(M[i].astype(np.int64, copy=False)),
                }

def simple_collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    
    return out