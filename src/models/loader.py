"""Unified model loader. `quant` is one of: none | int8 | nf4."""
from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

Quant = Literal["none", "int8", "nf4"]

_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _bnb_config(quant: Quant) -> BitsAndBytesConfig | None:
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_model(
    name: str,
    quant: Quant = "none",
    dtype: str = "float16",
    device_map: str | dict = "auto",
):
    bnb = _bnb_config(quant)
    kwargs: dict = {"device_map": device_map}
    if bnb is not None:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = _DTYPE[dtype]
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
