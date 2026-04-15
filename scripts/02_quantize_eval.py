"""Phase 2: PTQ evaluation (int8 / nf4) via bitsandbytes.

Usage:
    python scripts/02_quantize_eval.py --role student --quant nf4
    python scripts/02_quantize_eval.py --role teacher --quant int8
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
    ap.add_argument("--quant", choices=["int8", "nf4"], required=True)
    ap.add_argument("--model-path", default=None,
                    help="Override with a local (e.g. distilled) checkpoint.")
    ap.add_argument("--models-cfg", default="configs/models.yaml")
    ap.add_argument("--eval-cfg", default="configs/eval.yaml")
    ap.add_argument("--distilled", action="store_true",
                    help="Marks this run as using a distilled student.")
    args = ap.parse_args()

    mcfg = load_yaml(args.models_cfg)
    ecfg = load_yaml(args.eval_cfg)

    name = args.model_path or mcfg[args.role]["name"]
    tok = load_tokenizer(mcfg["tokenizer"]["name"])
    reset_peak()
    model = load_model(name, quant=args.quant)

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

    tag = f"{'distill+' if args.distilled else ''}ptq-{args.quant}-{args.role}"
    ctx = RunContext(
        tag=tag,
        model=name,
        quantization=args.quant,
        distilled=args.distilled,
        perplexity=ppl,
        tokens_per_sec=lat["tokens_per_sec"],
        ttft_ms=lat["ttft_ms"],
        peak_vram_gb=peak_vram_gb(),
    )
    run = new_run_dir(ctx.tag)
    out = save_metrics(run, ctx.to_dict())
    print(f"[done] {tag}  PPL={ppl:.3f}  tok/s={lat['tokens_per_sec']:.1f}  vram={peak_vram_gb():.2f}GB")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
