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
src/            Reusable modules
scripts/        CLI entry points, one per phase
modal_app.py    Modal entrypoint — runs each phase on a hosted GPU
notebooks/      Colab fallback (run.ipynb)
results/runs/   Per-run metrics.json (tracked in git)
checkpoints/    Model checkpoints (gitignored)
plots/          Generated figures
```

## Recommended workflow: Modal

Modal gives stable GPUs (no Colab disconnect) and runs scripts directly.

**One-time setup** (locally):
```bash
pip install modal
modal token new                                    # auth
modal secret create hf-token HF_TOKEN=hf_xxx       # or via the web UI
```

**Run a phase**:
```bash
modal run modal_app.py::baseline_eval --role student
modal run modal_app.py::baseline_eval --role teacher
modal run modal_app.py::quantize_eval --role student --quant nf4
modal run modal_app.py::distill_train
modal run modal_app.py::combined_eval --path checkpoints/distill-llama32-1b --quant nf4
modal run modal_app.py::plot_pareto
modal run modal_app.py::profile_run --path checkpoints/distill-llama32-1b --quant nf4
```

Phase 1/2/4/5 default to `L4`; Phase 3 (`distill_train`) uses `A100-40GB`.

**Pull results back locally** (so plots and git commits work):
```bash
modal volume get llama-results / ./results/
```

**Inspect runs without downloading**:
```bash
modal run modal_app.py::list_runs
```

## Phases

1. Baseline PPL + latency for 8B and 1B (no compression).
2. PTQ with bitsandbytes (int8, nf4) — measure PPL, VRAM, tokens/sec.
3. Logits distillation: 8B teacher → 1B student.
4. Combined: distilled-1B + nf4. Plot Pareto frontier.
5. `torch.profiler` — compute vs memory-movement breakdown.

## Colab fallback

If you can't or don't want to use Modal, `notebooks/run.ipynb` does the same
phases by `!python scripts/<phase>.py` from a Colab session. See the notebook
for details.
