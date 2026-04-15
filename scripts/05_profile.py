"""Phase 5: torch.profiler — compute vs memory-movement breakdown during decode.

Outputs a chrome trace + `top_ops.txt` under results/runs/<ts>-profile/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_model, load_tokenizer
from src.profile.torch_profiler import profile_generate
from src.utils import load_yaml, new_run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="model id or local path")
    ap.add_argument("--quant", choices=["none", "int8", "nf4"], default="none")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--prompt-length", type=int, default=512)
    ap.add_argument("--models-cfg", default="configs/models.yaml")
    args = ap.parse_args()

    mcfg = load_yaml(args.models_cfg)
    tok = load_tokenizer(mcfg["tokenizer"]["name"])
    model = load_model(args.path, quant=args.quant)

    device = next(model.parameters()).device
    input_ids = torch.randint(0, tok.vocab_size, (1, args.prompt_length), device=device)

    run = new_run_dir(f"profile-{args.quant}")
    out = profile_generate(model, input_ids, run / "trace", new_tokens=args.new_tokens)
    print(f"[saved] profiler traces -> {out}")


if __name__ == "__main__":
    main()
