"""Hinton-style logits distillation loss.

L = alpha * T^2 * KL(softmax(s/T) || softmax(t/T)) + (1-alpha) * CE(s, labels)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,   # (B, L, V)
    teacher_logits: torch.Tensor,   # (B, L, V)
    labels: torch.Tensor,           # (B, L)
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, dict]:
    # Shift for causal LM: predict token t+1 from positions 0..L-2.
    s = student_logits[..., :-1, :].contiguous()
    t = teacher_logits[..., :-1, :].contiguous()
    y = labels[..., 1:].contiguous()

    # KL per token, averaged over valid positions.
    s_log = F.log_softmax(s / temperature, dim=-1)
    t_prob = F.softmax(t / temperature, dim=-1)
    kl = F.kl_div(s_log, t_prob, reduction="none").sum(dim=-1)  # (B, L-1)
    mask = (y != -100).float()
    kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)

    ce = F.cross_entropy(s.view(-1, s.size(-1)), y.view(-1), ignore_index=-100)

    loss = alpha * (temperature**2) * kl + (1.0 - alpha) * ce
    return loss, {"loss_kd": kl.detach(), "loss_ce": ce.detach()}
