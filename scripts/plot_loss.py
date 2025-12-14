"""
Plot training loss from common outputs in this repo.

Supports:
- HF Trainer logs: runs/.../checkpoint-*/trainer_state.json
- Console logs captured with tee: runs/.../train.log
- CSV loss file: runs/.../loss.csv

Examples:
  python scripts/plot_loss.py --run_dir runs/qwen3_0p6b_qat2b
  python scripts/plot_loss.py --run_dir runs/qwen3_0p6b_qat2b --source trainer
  python scripts/plot_loss.py --run_dir runs/qwen3_0p6b_qat2b --save runs/qwen3_0p6b_qat2b/loss.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import List, Tuple


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting.\n"
            "Install it with: uv pip install matplotlib\n"
            f"Import error: {e}"
        )
    return plt


def _configure_y_grid(ax, step: float):
    from matplotlib.ticker import MultipleLocator  # type: ignore

    ax.yaxis.set_minor_locator(MultipleLocator(step))
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.8, alpha=0.25)
    ax.grid(which="major", axis="y", linestyle="--", linewidth=1.0, alpha=0.25)


def _configure_y_grid_major_minor(ax, *, major_step: float, minor_step: float):
    from matplotlib.ticker import MultipleLocator  # type: ignore

    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
    # Major lines: every 1.0 (solid)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=1.0, alpha=0.25)
    # Minor lines: every 0.5 (more visible dotted)
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=1.1, alpha=0.35)


def _parse_trainer_state(run_dir: Path) -> Tuple[List[float], List[float], str]:
    paths = sorted(glob.glob(str(run_dir / "checkpoint-*" / "trainer_state.json")))
    if not paths:
        raise FileNotFoundError(f"No trainer_state.json found under {run_dir}/checkpoint-*/")

    # Pick the latest checkpoint by numeric suffix if possible.
    def key(p: str) -> int:
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else 0

    state_path = max(paths, key=key)
    state = json.load(open(state_path, "r"))

    xs: List[float] = []
    ys: List[float] = []
    for row in state.get("log_history", []):
        if "loss" in row and "step" in row:
            xs.append(float(row["step"]))
            ys.append(float(row["loss"]))

    if not xs:
        raise RuntimeError(f"Found {state_path} but no (step, loss) in log_history.")

    title = f"Loss (HF Trainer) — {Path(state_path).parent.name}"
    return xs, ys, title


def _parse_train_log(run_dir: Path) -> Tuple[List[float], List[float], str]:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {log_path}. Create it via: ... 2>&1 | tee {log_path}")

    pat = re.compile(r"opt_step=(\d+).*?loss=([0-9.]+)")
    xs: List[float] = []
    ys: List[float] = []
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if m:
            xs.append(float(m.group(1)))
            ys.append(float(m.group(2)))
    if not xs:
        raise RuntimeError(f"Found {log_path} but did not match any opt_step/loss lines.")

    return xs, ys, "Loss (train.log)"


def _parse_loss_csv(run_dir: Path) -> Tuple[List[float], List[float], str]:
    # Prefer loss.csv, but also support rotated files: loss_prev_*.csv
    csv_path = run_dir / "loss.csv"
    if not csv_path.exists():
        candidates = sorted(run_dir.glob("loss_prev_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Missing {csv_path}.")
        csv_path = candidates[-1]

    xs: List[float] = []
    ys: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "step" in row and "loss" in row and row["step"] and row["loss"]:
                xs.append(float(row["step"]))
                ys.append(float(row["loss"]))
    if not xs:
        raise RuntimeError(f"Found {csv_path} but it had no rows.")

    return xs, ys, f"Loss ({csv_path.name})"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="runs/qwen3_0p6b_qat2b")
    p.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "trainer", "trainlog", "csv"],
        help="Which log format to use. auto tries csv -> trainlog -> trainer.",
    )
    p.add_argument("--save", type=str, default=None, help="Optional path to save a PNG.")
    p.add_argument("--no_show", action="store_true", help="Do not open a GUI window.")
    p.add_argument(
        "--ygrid_step",
        type=float,
        default=0.1,
        help="Horizontal grid spacing on the loss axis (minor grid, dotted).",
    )
    p.add_argument(
        "--y_major_step",
        type=float,
        default=1.0,
        help="Major y-grid/tick spacing (solid).",
    )
    p.add_argument(
        "--y_minor_step",
        type=float,
        default=0.5,
        help="Minor y-grid spacing (dotted).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    sources = [args.source] if args.source != "auto" else ["csv", "trainlog", "trainer"]
    last_err: Exception | None = None

    for src in sources:
        try:
            if src == "trainer":
                xs, ys, title = _parse_trainer_state(run_dir)
            elif src == "trainlog":
                xs, ys, title = _parse_train_log(run_dir)
            elif src == "csv":
                xs, ys, title = _parse_loss_csv(run_dir)
            else:
                raise ValueError(f"Unknown source: {src}")
            break
        except Exception as e:
            last_err = e
    else:
        raise SystemExit(f"Could not find any loss source under {run_dir}. Last error: {last_err}")

    plt = _require_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)

    # Y-grid: major every 1.0 (solid), minor every 0.5 (dotted).
    if args.y_major_step and args.y_major_step > 0 and args.y_minor_step and args.y_minor_step > 0:
        _configure_y_grid_major_minor(ax, major_step=args.y_major_step, minor_step=args.y_minor_step)
    elif args.ygrid_step and args.ygrid_step > 0:
        # Backwards-compatible single-step minor grid.
        _configure_y_grid(ax, args.ygrid_step)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
