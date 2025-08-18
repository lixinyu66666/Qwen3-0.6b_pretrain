"""
Parse training logs and plot Loss vs Steps.

Supported line pattern example:
  step   28550 | loss 4.4506 | tokens/step ~ 24576

Usage:
  python scripts/plot_loss.py --log train.log --out loss.png --csv loss.csv --ma 100
  # If your logs are in nohup.out:
  python scripts/plot_loss.py --log nohup.out --out loss.png

Options:
  --ma N       Moving-average window size (integer > 1). Mutually exclusive with --ema.
  --ema A      Exponential moving average, decay in (0,1). Mutually exclusive with --ma.
  --csv PATH   Save parsed (step, loss) pairs to CSV.
  --every K    Downsample: keep one point every K steps (default 1 = keep all).
  --xlim A B   Optional x-axis limits (steps).
  --ylim A B   Optional y-axis limits (loss).
"""

import os, sys
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils.plot_loss import read_steps_losses, moving_average, ema, downsample

def parse_args():
    ap = argparse.ArgumentParser(description="Plot training loss from log file.")
    ap.add_argument("--log", required=True, help="Path to the training log file.")
    ap.add_argument("--out", default="results/loss.png", help="Output image file for the plot.")
    ap.add_argument("--csv", default=None, help="Optional CSV file to save (step, loss) pairs.")
    ap.add_argument("--ma", type=int, default=0, help="Moving average window size (integer > 1).")
    ap.add_argument("--ema", type=float, default=None, help="Exponential moving average decay in (0,1).")
    ap.add_argument("--every", type=int, default=1, help="Downsample: keep one point every K steps.")
    ap.add_argument("--xlim", nargs=2, type=int, default=None, help="Optional x-axis limits (steps).")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, help="Optional y-axis limits (loss).")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.ma and args.ema is not None:
        print("'--ma' and '--ema' are mutually exclusive. EMA will be ignored.")
        args.ema = None
    
    steps, losses = read_steps_losses(args.log)
    if not steps:
        print(f"No valid (step, loss) pairs found in {args.log}.")
        return

    steps, losses = downsample(steps, losses, args.every)

    y_plot = losses[:]
    label = "Loss"
    if args.ma and args.ma > 1:
        y_plot = moving_average(losses, args.ma)
        label += f" (MA {args.ma})"
    elif args.ema is not None:
        a = float(args.ema)
        if 0.0 < a < 1.0:
            y_plot = ema(losses, a)
            label += f" (EMA {a})"
        else:
            print(f"Invalid EMA decay {a}. Must be in (0,1).")
    
    plt.figure(figsize=(9, 5))
    plt.plot(steps, y_plot, linewidth=1.2)
    plt.xlabel("Steps")
    plt.ylabel(label)
    plt.title("Training Loss vs Steps")
    plt.grid(True)
    if args.xlim:
        plt.xlim(args.xlim[0], args.xlim[1])
    if args.ylim:
        plt.ylim(args.ylim[0], args.ylim[1])
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Plot saved to {args.out}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("step,loss\n")
            for step, loss in zip(steps, losses):
                f.write(f"{step},{loss}\n")
        print(f"CSV saved to {args.csv}")

if __name__ == "__main__":
    main()