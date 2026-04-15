"""Inference latency: TTFT (prefill) + decode tokens/sec."""
from __future__ import annotations

import time

import torch


@torch.no_grad()
def measure_latency(
    model,
    tokenizer,
    prompt_length: int = 512,
    generate_new_tokens: int = 128,
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> dict:
    device = next(model.parameters()).device
    # Construct a synthetic prompt of the requested token length.
    vocab = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab, (1, prompt_length), device=device)
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs = dict(
        max_new_tokens=generate_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    for _ in range(warmup_runs):
        model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

    ttfts, decode_rates = [], []
    for _ in range(timed_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # TTFT: one prefill forward pass.
        _ = model(input_ids, attention_mask=attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        t1 = time.perf_counter()
        out = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total = time.perf_counter() - t1

        new_tokens = out.shape[1] - input_ids.shape[1]
        decode_time = max(total - ttft, 1e-6)
        ttfts.append(ttft * 1000)
        decode_rates.append(new_tokens / decode_time)

    return {
        "ttft_ms": sum(ttfts) / len(ttfts),
        "tokens_per_sec": sum(decode_rates) / len(decode_rates),
        "prompt_length": prompt_length,
        "generate_new_tokens": generate_new_tokens,
    }
