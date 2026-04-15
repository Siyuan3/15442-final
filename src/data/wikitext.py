"""WikiText-2 loaders for PPL evaluation and distillation training."""
from __future__ import annotations

from datasets import load_dataset


def load_wikitext_text(split: str = "test", subset: str = "wikitext-2-raw-v1") -> str:
    """Concatenate the raw text — used by the sliding-window PPL loop."""
    ds = load_dataset(subset, split=split) if "/" not in subset else load_dataset("wikitext", subset, split=split)
    # datasets>=2.14 prefers (path, name) form:
    if "text" not in ds.column_names:
        ds = load_dataset("wikitext", subset, split=split)
    return "\n\n".join(t for t in ds["text"] if t.strip())


def load_wikitext_blocks(tokenizer, block_size: int = 1024, split: str = "train",
                         subset: str = "wikitext-2-raw-v1"):
    """Tokenize train set and pack into fixed-length blocks for causal LM training."""
    ds = load_dataset("wikitext", subset, split=split)

    def tok_fn(batch):
        return tokenizer(batch["text"])

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    def group(examples):
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concat["input_ids"]) // block_size) * block_size
        result = {
            k: [v[i : i + block_size] for i in range(0, total, block_size)]
            for k, v in concat.items()
        }
        result["labels"] = [ids.copy() for ids in result["input_ids"]]
        return result

    return tokenized.map(group, batched=True)
