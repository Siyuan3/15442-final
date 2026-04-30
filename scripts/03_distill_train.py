"""Phase 3: Logits distillation from teacher (8B) to student (1B)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import DataCollatorForLanguageModeling, TrainingArguments

from src.data.wikitext import load_wikitext_blocks
from src.distill.trainer import DistillationTrainer
from src.models.loader import load_model, load_tokenizer
from src.utils import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-cfg", default="configs/models.yaml")
    ap.add_argument("--distill-cfg", default="configs/distill.yaml")
    args = ap.parse_args()

    mcfg = load_yaml(args.models_cfg)
    dcfg = load_yaml(args.distill_cfg)

    tok = load_tokenizer(mcfg["tokenizer"]["name"])

    # Teacher frozen, optionally 8-bit to save VRAM.
    teacher = load_model(
        mcfg["teacher"]["name"],
        quant="int8" if dcfg["train"].get("teacher_load_in_8bit", False) else "none",
        dtype=mcfg["teacher"]["dtype"],
    )
    student = load_model(
        mcfg["student"]["name"], quant="none", dtype=mcfg["student"]["dtype"],
    )
    if dcfg["train"].get("gradient_checkpointing"):
        student.gradient_checkpointing_enable()
        student.config.use_cache = False

    train_ds = load_wikitext_blocks(
        tok,
        block_size=dcfg["data"]["block_size"],
        split=dcfg["data"]["split"],
        subset=dcfg["data"]["subset"],
    )

    t = dcfg["train"]
    ta_kwargs = dict(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        gradient_checkpointing=t["gradient_checkpointing"],
        report_to=[],
        remove_unused_columns=False,
    )
    if "max_steps" in t:
        ta_kwargs["max_steps"] = t["max_steps"]
    elif "num_train_epochs" in t:
        ta_kwargs["num_train_epochs"] = t["num_train_epochs"]
    training_args = TrainingArguments(**ta_kwargs)

    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tok,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        kd_temperature=dcfg["loss"]["temperature"],
        kd_alpha=dcfg["loss"]["alpha_kd"],
    )
    trainer.train()
    trainer.save_model(t["output_dir"])
    tok.save_pretrained(t["output_dir"])
    print(f"[done] saved distilled student -> {t['output_dir']}")


if __name__ == "__main__":
    main()
