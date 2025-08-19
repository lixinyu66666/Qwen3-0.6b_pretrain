# check_safetensors_shapes.py
from safetensors import safe_open
import sys

path = sys.argv[1]
with safe_open(path, framework="pt", device="cpu") as f:
    bad = []
    for k in f.keys():
        shape = f.get_tensor(k).shape
        if len(shape) == 1 and shape[0] == 0:
            bad.append(k)
    if bad:
        print("[!] Found zero-sized tensors:", len(bad))
        for k in bad[:10]:
            print("   ", k, "[0]")
        print("=> This checkpoint is broken (saved without ZeRO-3 16bit gather).")
    else:
        print("[OK] No zero-sized tensors detected.")