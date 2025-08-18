import re
from typing import List, Tuple, Optional
from collections import OrderedDict, deque

def build_regex() -> re.Pattern:
    """
    Build a robust regex to match 'step ... | loss ...'.
    - 'step' and 'loss' are matched case-insensitively
    - step: integer
    - loss: float (supports scientific notation)
    """
    return re.compile(
        r"""(?i)              # case-insensitive
            \bstep\b\s*      # 'step' + spaces
            (?P<step>\d+)    # integer step
            .*?              # anything (non-greedy) until 'loss'
            \bloss\b\s*      # 'loss' + spaces
            (?P<loss>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) # float/scientific
        """,
        re.VERBOSE,
    )

def read_steps_losses(log_file: str) -> Tuple[list[int], list[float]]:
    """
    Stream-parse lines from the log file and extract (step, loss).
    For duplicate steps, keep the last occurrence.
    """
    pat = build_regex()
    last = OrderedDict()  # step -> loss
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            step = int(m.group("step"))
            loss = float(m.group("loss"))
            last[step] = loss  # keep the last occurrence for each step
    steps = list(last.keys())
    losses = [last[s] for s in steps]
    return steps, losses

def moving_average(y: List[float], k: int) -> List[float]:
    if k <= 1 or len(y) <= 1:
        return y[:]
    out, s = [], 0.0
    dq = deque()
    for v in y:
        dq.append(v)
        s += v
        if len(dq) > k:
            s -= dq.popleft()
        out.append(s / len(dq))
    return out

def ema(y: List[float], alpha: float) -> List[float]:
    """Exponential moving average with decay alpha in (0,1)."""
    n = len(y)
    if n == 0:
        return []
    out = [y[0]]
    for i in range(1, n):
        out.append(alpha * out[-1] + (1.0 - alpha) * y[i])
    
    return out

def downsample(steps: List[int], losses: List[float], k: int) -> Tuple[List[int], List[float]]:
    """Keep one point every k samples."""
    if k <= 1:
        return steps, losses
    return steps[::k], losses[::k]