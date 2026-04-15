"""HF Trainer subclass that injects a frozen teacher and swaps in the KD loss."""
from __future__ import annotations

import torch
from transformers import Trainer

from .losses import kd_loss


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, kd_temperature: float = 2.0,
                 kd_alpha: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher_model is not None, "teacher_model is required"
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        student_out = model(**{k: v for k, v in inputs.items() if k != "labels"},
                            labels=None)
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        loss, logs = kd_loss(
            student_out.logits, teacher_out.logits, labels,
            temperature=self.kd_temperature, alpha=self.kd_alpha,
        )
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({k: float(v) for k, v in logs.items()})
        return (loss, student_out) if return_outputs else loss
