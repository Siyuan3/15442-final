"""GSM8K loader for reasoning eval (used in later phases)."""
from __future__ import annotations

from datasets import load_dataset


def load_gsm8k(split: str = "test", num_examples: int | None = None):
    ds = load_dataset("gsm8k", "main", split=split)
    if num_examples is not None:
        ds = ds.select(range(min(num_examples, len(ds))))
    return ds


def extract_answer(text: str) -> str | None:
    """GSM8K answers are after '#### '."""
    if "####" not in text:
        return None
    return text.split("####")[-1].strip().replace(",", "").replace("$", "")
