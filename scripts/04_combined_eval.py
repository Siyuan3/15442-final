"""Phase 4: Evaluate distilled student (optionally quantized) + aggregate all
runs into a Pareto-frontier plot.

Examples:
    # distilled-only (fp16)
    python scripts/04_combined_eval.py eval --path checkpoints/distill-llama32-1b

    # distilled + 4-bit
    python scripts/04_combined_eval.py eval --path checkpoints/distill-llama32-1b --quant nf4

    # aggregate all runs and plot
    python scripts/04_combined_eval.py plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Light import for the `plot` subcommand. Heavy deps (torch, transformers,
# datasets) are imported inside cmd_eval so plotting works locally without
# them.
from src.utils import REPO_ROOT, RunContext, load_yaml, new_run_dir, save_metrics


def cmd_eval(args):
    from src.data.wikitext import load_wikitext_text
    from src.eval.latency import measure_latency
    from src.eval.memory import peak_vram_gb, reset_peak
    from src.eval.perplexity import compute_perplexity
    from src.models.loader import load_model, load_tokenizer

    mcfg = load_yaml(args.models_cfg)
    ecfg = load_yaml(args.eval_cfg)
    tok = load_tokenizer(mcfg["tokenizer"]["name"])
    reset_peak()
    model = load_model(args.path, quant=args.quant, dtype=mcfg["student"]["dtype"])

    text = load_wikitext_text(
        split=ecfg["perplexity"]["split"],
        subset=ecfg["perplexity"]["subset"],
    )
    ppl = compute_perplexity(model, tok, text,
                             block_size=ecfg["perplexity"]["block_size"],
                             stride=ecfg["perplexity"]["stride"])
    lat = measure_latency(model, tok, **ecfg["latency"])

    tag = f"distilled-{args.quant}"
    ctx = RunContext(
        tag=tag, model=args.path, quantization=args.quant, distilled=True,
        perplexity=ppl, tokens_per_sec=lat["tokens_per_sec"],
        ttft_ms=lat["ttft_ms"], peak_vram_gb=peak_vram_gb(),
    )
    run = new_run_dir(tag)
    save_metrics(run, ctx.to_dict())
    print(f"[done] {tag}  PPL={ppl:.3f}  tok/s={lat['tokens_per_sec']:.1f}")


def cmd_plot(args):
    import matplotlib.pyplot as plt

    runs_dir = REPO_ROOT / "results" / "runs"
    records = []
    for metrics_file in sorted(runs_dir.glob("*/metrics.json")):
        with open(metrics_file) as f:
            records.append(json.load(f))
    if not records:
        print("no runs found under results/runs/")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in records:
        if r.get("tokens_per_sec") is None or r.get("perplexity") is None:
            continue
        marker = "o" if r["distilled"] else "s"
        color = {"none": "tab:blue", "int8": "tab:orange", "nf4": "tab:red"}.get(
            r["quantization"], "gray")
        ax.scatter(r["tokens_per_sec"], r["perplexity"], marker=marker, c=color, s=80)
        ax.annotate(r["tag"], (r["tokens_per_sec"], r["perplexity"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Decode speed (tokens/sec)")
    ax.set_ylabel("WikiText-2 Perplexity (lower is better)")
    ax.set_title("Pareto: Perplexity vs Latency")
    ax.grid(alpha=0.3)
    # Write into results/ so the persistent Modal volume captures it.
    out = REPO_ROOT / "results" / "plots" / "pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval")
    pe.add_argument("--path", required=True, help="distilled checkpoint dir")
    pe.add_argument("--quant", choices=["none", "int8", "nf4"], default="none")
    pe.add_argument("--models-cfg", default="configs/models.yaml")
    pe.add_argument("--eval-cfg", default="configs/eval.yaml")
    pe.set_defaults(fn=cmd_eval)

    pp = sub.add_parser("plot")
    pp.set_defaults(fn=cmd_plot)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
