"""Aggregate batch-sweep runs under results/runs/*-batchsweep-* and
render batch-vs-throughput curves for each (role, quant) configuration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import REPO_ROOT


def main():
    import matplotlib.pyplot as plt

    runs_dir = REPO_ROOT / "results" / "runs"
    sweeps = []
    for f in sorted(runs_dir.glob("*batchsweep*/metrics.json")):
        with open(f) as fp:
            sweeps.append(json.load(fp))
    if not sweeps:
        print("no batch-sweep runs found under results/runs/*batchsweep*/")
        return

    color = {"none": "tab:blue", "int8": "tab:orange", "nf4": "tab:red"}
    style = {"student": "-o", "teacher": "--s"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ylabel, title in [
        (axes[0], "tokens_per_sec_per_seq",
         "Per-sequence tok/s",
         "Per-Sequence Throughput vs Batch Size"),
        (axes[1], "tokens_per_sec_total",
         "Total throughput (tok/s)",
         "Total Throughput vs Batch Size"),
    ]:
        for s in sweeps:
            xs = [r["batch_size"] for r in s["results"]]
            ys = [r[key] for r in s["results"]]
            label = f"{s['role']}-{s['quantization']}"
            ax.plot(xs, ys, style[s["role"]], color=color[s["quantization"]],
                    label=label, linewidth=2, markersize=8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    out = REPO_ROOT / "results" / "plots" / "batch_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
