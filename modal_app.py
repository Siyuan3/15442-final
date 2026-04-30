"""Modal entry point — runs each phase on a hosted GPU.

Local source (src/, scripts/, configs/) is bind-mounted into /root.
Results, checkpoints, and the HF cache live in named Modal Volumes so
they accumulate across runs and survive function restarts.

Setup once (locally):
    pip install modal
    modal token new
    modal secret create hf-token HF_TOKEN=hf_xxx   # or via the web UI

Run a phase (from repo root):
    modal run modal_app.py::baseline_eval --role student
    modal run modal_app.py::baseline_eval --role teacher
    modal run modal_app.py::quantize_eval --role student --quant nf4
    modal run modal_app.py::quantize_eval --role student --quant int8
    modal run modal_app.py::quantize_eval --role teacher --quant nf4
    modal run modal_app.py::quantize_eval --role teacher --quant int8
    modal run modal_app.py::distill_train
    modal run modal_app.py::combined_eval --path checkpoints/distill-llama32-1b --quant nf4
    modal run modal_app.py::plot_pareto
    modal run modal_app.py::profile_run --path checkpoints/distill-llama32-1b --quant nf4

Pull results back to the local repo so plots / git-commits work locally:
    modal volume get llama-results / ./results/

(Modal `volume get` writes to the destination directory — it preserves the
volume's internal layout, so files land at ./results/runs/<ts>-<tag>/...)
"""
from __future__ import annotations

import modal

APP_NAME = "kd-ptq-llama"

# ---------------------------------------------------------------------------
# Image: PyTorch w/ CUDA + project requirements + local source.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("configs", remote_path="/root/configs")
)

# ---------------------------------------------------------------------------
# Persistent storage. Create-if-missing; named so multiple runs accumulate.
# ---------------------------------------------------------------------------
results_vol = modal.Volume.from_name("llama-results", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("llama-checkpoints", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("llama-hf-cache", create_if_missing=True)

VOLUMES = {
    "/root/results": results_vol,
    "/root/checkpoints": checkpoints_vol,
    "/cache/hf": hf_cache_vol,
}
SECRETS = [modal.Secret.from_name("hf-token")]

# GPU choice. Use the same GPU for every measurement so the Pareto plot is
# directly comparable; A100-40GB is the smallest unit that fits the
# distillation memory footprint (8B teacher + 1B student + grads + optim).
EVAL_GPU = "A100-40GB"   # all eval / latency measurements
TRAIN_GPU = "A100-40GB"  # distillation training

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Helpers (run inside container).
# ---------------------------------------------------------------------------
def _setup_env() -> None:
    import os
    import sys

    os.environ["HF_HOME"] = "/cache/hf"
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    os.chdir("/root")


def _run(args: list[str]) -> None:
    """Invoke a project script and persist any volume writes."""
    import subprocess

    subprocess.run(["python", "-u", *args], cwd="/root", check=True)
    results_vol.commit()
    checkpoints_vol.commit()
    hf_cache_vol.commit()


# ---------------------------------------------------------------------------
# Phase 1 — Baseline (uncompressed teacher / student).
# ---------------------------------------------------------------------------
@app.function(gpu=EVAL_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def baseline_eval(role: str = "student") -> None:
    _setup_env()
    _run(["scripts/01_baseline_eval.py", "--role", role])


# ---------------------------------------------------------------------------
# Phase 2 — Post-Training Quantization (bitsandbytes).
# ---------------------------------------------------------------------------
@app.function(gpu=EVAL_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def quantize_eval(role: str = "student", quant: str = "nf4") -> None:
    _setup_env()
    _run(["scripts/02_quantize_eval.py", "--role", role, "--quant", quant])


# ---------------------------------------------------------------------------
# Phase 3 — Knowledge Distillation (8B → 1B).
# ---------------------------------------------------------------------------
@app.function(gpu=TRAIN_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=14400)
def distill_train() -> None:
    _setup_env()
    _run(["scripts/03_distill_train.py"])


# ---------------------------------------------------------------------------
# Phase 4 — Combined eval + Pareto plot.
# ---------------------------------------------------------------------------
@app.function(gpu=EVAL_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def combined_eval(path: str, quant: str = "nf4") -> None:
    """`path` is a path INSIDE the container (e.g. checkpoints/distill-llama32-1b)."""
    _setup_env()
    _run(["scripts/04_combined_eval.py", "eval", "--path", path, "--quant", quant])


@app.function(volumes=VOLUMES, timeout=600)
def plot_pareto() -> None:
    _setup_env()
    _run(["scripts/04_combined_eval.py", "plot"])


# ---------------------------------------------------------------------------
# Phase 5 — torch.profiler trace.
# ---------------------------------------------------------------------------
@app.function(gpu=EVAL_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def profile_run(path: str, quant: str = "nf4") -> None:
    _setup_env()
    _run(["scripts/05_profile.py", "--path", path, "--quant", quant])


# ---------------------------------------------------------------------------
# Extra phase — batch-size sweep for one (role, quant) configuration.
# ---------------------------------------------------------------------------
@app.function(gpu=EVAL_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def batch_sweep(role: str = "student", quant: str = "none") -> None:
    _setup_env()
    _run(["scripts/06_batch_sweep.py", "--role", role, "--quant", quant])


# ---------------------------------------------------------------------------
# Convenience: dump the metrics.json contents from the volume to stdout.
# ---------------------------------------------------------------------------
@app.function(volumes=VOLUMES, timeout=120)
def list_runs() -> None:
    import json
    import os

    root = "/root/results/runs"
    if not os.path.isdir(root):
        print("(no runs yet)")
        return
    for name in sorted(os.listdir(root)):
        f = os.path.join(root, name, "metrics.json")
        if not os.path.isfile(f):
            continue
        with open(f) as fp:
            m = json.load(fp)
        print(f"{name}: PPL={m.get('perplexity')}  tok/s={m.get('tokens_per_sec')}  "
              f"vram={m.get('peak_vram_gb')}  quant={m.get('quantization')}")


@app.local_entrypoint()
def main() -> None:
    """`modal run modal_app.py` — picks a sensible default. Use `::<fn>` for the rest."""
    baseline_eval.remote(role="student")
