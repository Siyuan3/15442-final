"""Phase 1: Baseline PPL + latency for teacher and student (no compression).

Usage:
    python scripts/01_baseline_eval.py --role student
    python scripts/01_baseline_eval.py --role teacher
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wikitext import load_wikitext_text
from src.eval.latency import measure_latency
from src.eval.memory import peak_vram_gb, reset_peak
from src.eval.perplexity import compute_perplexity
from src.models.loader import load_model, load_tokenizer
from src.utils import RunContext, load_yaml, new_run_dir, save_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["teacher", "student"], required=True)
    ap.add_argument("--models-cfg", default="configs/models.yaml")
    ap.add_argument("--eval-cfg", default="configs/eval.yaml")
    args = ap.parse_args()

    mcfg = load_yaml(args.models_cfg)
    ecfg = load_yaml(args.eval_cfg)

    model_cfg = mcfg[args.role]
    tok = load_tokenizer(mcfg["tokenizer"]["name"])
    reset_peak()
    model = load_model(model_cfg["name"], quant="none", dtype=model_cfg["dtype"])

    text = load_wikitext_text(
        split=ecfg["perplexity"]["split"],
        subset=ecfg["perplexity"]["subset"],
    )
    ppl = compute_perplexity(
        model, tok, text,
        block_size=ecfg["perplexity"]["block_size"],
        stride=ecfg["perplexity"]["stride"],
    )
    lat = measure_latency(model, tok, **ecfg["latency"])

    ctx = RunContext(
        tag=f"baseline-{args.role}",
        model=model_cfg["name"],
        quantization="none",
        distilled=False,
        perplexity=ppl,
        tokens_per_sec=lat["tokens_per_sec"],
        ttft_ms=lat["ttft_ms"],
        peak_vram_gb=peak_vram_gb(),
        extra={"latency_cfg": ecfg["latency"]},
    )
    run = new_run_dir(ctx.tag)
    out = save_metrics(run, ctx.to_dict())
    print(f"[done] PPL={ppl:.3f}  tok/s={lat['tokens_per_sec']:.1f}  vram={peak_vram_gb():.2f}GB")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
