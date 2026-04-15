# Efficient LLM Inference: Knowledge Distillation × Quantization

CMU 15-442 course project. Studies the synergy of Knowledge Distillation (KD)
and Post-Training Quantization (PTQ) on Llama-3 models, plotting the
perplexity–latency Pareto frontier.

- **Teacher**: `meta-llama/Llama-3.1-8B`
- **Student**: `meta-llama/Llama-3.2-1B`
- **Eval datasets**: WikiText-2 (PPL), GSM8K (reasoning)

## Layout

```
configs/        YAML configs (models, distill, eval)
src/            Reusable modules — import from notebooks & scripts
scripts/        CLI entry points, one per phase
notebooks/      Colab notebooks (call into src/)
results/runs/   Per-run metrics.json (tracked in git)
checkpoints/    Model checkpoints (gitignored, store on Drive)
plots/          Generated figures
```

## Workflow

- **Local**: edit code in `src/`, commit, push.
- **Colab**: `git pull`, run the notebook for the current phase.
- Every script writes `results/runs/<timestamp>/metrics.json` with a fixed
  schema so `notebooks/04_pareto.ipynb` can aggregate without hand-copying.

## Phases

1. Baseline PPL + latency for 8B and 1B (no compression).
2. PTQ with bitsandbytes (8-bit, 4-bit) — measure PPL, VRAM, tokens/sec.
3. Logits distillation: 8B teacher → 1B student.
4. Combined: distilled-1B + 4-bit PTQ. Plot Pareto frontier.
5. `torch.profiler` — compute vs memory-movement breakdown.

## Setup

See `notebooks/00_setup.ipynb` for the Colab bootstrap (HF login, Drive mount,
deps). Locally: `pip install -r requirements.txt`.
