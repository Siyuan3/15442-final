"""Phase-extra: Batch-size sweep for one (model, quant) configuration.

Sweeps batch_size in [1, 2, 4, 8, 16, 32], measuring decode throughput.
Writes a single JSON under results/runs/<ts>-batchsweep-<tag>/metrics.json
with the per-batch throughput numbers, so the plot script can render
batch-vs-throughput curves across configs.

Examples:
    python scripts/06_batch_sweep.py --role student --quant none
    python scripts/06_batch_sweep.py --role student --quant nf4
    python scripts/06_batch_sweep.py --role teacher --quant none
    python scripts/06_batch_sweep.py --role teacher --quant nf4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.latency import measure_latency
from src.eval.memory import peak_vram_gb, reset_peak
from src.models.loader import load_model, load_tokenizer
from src.utils import load_yaml, new_run_dir


DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["teacher", "student"], required=True)
    ap.add_argument("--quant", choices=["none", "int8", "nf4"], required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    ap.add_argument("--prompt-length", type=int, default=256)
    ap.add_argument("--generate-new-tokens", type=int, default=64)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--timed-runs", type=int, default=3)
    ap.add_argument("--models-cfg", default="configs/models.yaml")
    args = ap.parse_args()

    mcfg = load_yaml(args.models_cfg)
    name = args.model_path or mcfg[args.role]["name"]
    tok = load_tokenizer(mcfg["tokenizer"]["name"])
    model = load_model(name, quant=args.quant, dtype=mcfg[args.role]["dtype"])

    results = []
    for bs in args.batches:
        reset_peak()
        try:
            lat = measure_latency(
                model, tok,
                prompt_length=args.prompt_length,
                generate_new_tokens=args.generate_new_tokens,
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
                batch_size=bs,
            )
        except Exception as e:
            print(f"[skip] batch={bs} failed: {e}")
            continue
        per_seq = lat["tokens_per_sec"]
        # tokens_per_sec returned by measure_latency is per-sequence; total
        # throughput across all sequences in the batch is bs * per_seq.
        total = per_seq * bs
        vram = peak_vram_gb()
        print(f"[batch={bs:>3}]  per_seq={per_seq:6.2f}  total={total:7.2f} tok/s  "
              f"vram={vram:5.2f} GB  ttft={lat['ttft_ms']:.1f} ms")
        results.append({
            "batch_size": bs,
            "tokens_per_sec_per_seq": per_seq,
            "tokens_per_sec_total": total,
            "ttft_ms": lat["ttft_ms"],
            "peak_vram_gb": vram,
        })

    tag = f"batchsweep-{args.quant}-{args.role}"
    run_dir = new_run_dir(tag)
    out = {
        "tag": tag,
        "model": name,
        "quantization": args.quant,
        "role": args.role,
        "prompt_length": args.prompt_length,
        "generate_new_tokens": args.generate_new_tokens,
        "results": results,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
