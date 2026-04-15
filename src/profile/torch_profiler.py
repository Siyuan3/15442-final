"""torch.profiler wrapper: captures a short decode window and dumps traces."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


def profile_generate(model, input_ids, out_dir: Path, new_tokens: int = 32):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(out_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            model.generate(
                input_ids,
                max_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=model.config.eos_token_id,
            )
            prof.step()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    (out_dir / "top_ops.txt").write_text(table)
    return out_dir
