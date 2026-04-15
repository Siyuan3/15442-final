"""Sliding-window perplexity on WikiText-2 — standard HF recipe."""
from __future__ import annotations

import math

import torch
from tqdm import tqdm


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    text: str,
    block_size: int = 1024,
    stride: int = 512,
    device: str | None = None,
) -> float:
    if device is None:
        device = next(model.parameters()).device
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    nll_sum, n_tokens = 0.0, 0
    prev_end = 0
    for begin in tqdm(range(0, seq_len, stride), desc="ppl"):
        end = min(begin + block_size, seq_len)
        trg_len = end - prev_end
        window = input_ids[:, begin:end].to(device)
        targets = window.clone()
        targets[:, :-trg_len] = -100

        out = model(window, labels=targets)
        # HF averages over non-ignored tokens; multiply back by count.
        num_valid = (targets != -100).sum().item() - 1  # shift-by-one for CLM
        if num_valid <= 0:
            prev_end = end
            continue
        nll_sum += out.loss.float().item() * num_valid
        n_tokens += num_valid
        prev_end = end
        if end == seq_len:
            break

    return math.exp(nll_sum / n_tokens)
