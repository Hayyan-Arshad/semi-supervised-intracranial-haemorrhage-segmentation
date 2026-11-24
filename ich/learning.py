import math

import torch
from torch.nn import functional as F


def ramp(step, length):
    # Same exponential ramp schedule used by SSL4MIS / Mean Teacher.
    return math.exp(-5 * (1 - min(step / max(length, 1), 1)) ** 2)


@torch.no_grad()
def update_ema(student, teacher, step, decay=.99):
    alpha = min(decay, 1 - 1 / (step + 1))
    source_parameters = dict(student.named_parameters())
    for name, target in teacher.named_parameters():
        target.lerp_(source_parameters[name], 1 - alpha)
    # Copy buffers rather than silently leaving BN statistics stale.
    source_buffers = dict(student.named_buffers())
    for name, target in teacher.named_buffers():
        if name in source_buffers:
            target.copy_(source_buffers[name])
    # TinyViT caches attention biases in eval mode. Rebuild after EMA updates.
    teacher.eval()


def supervised_loss(logits, mask):
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probability * mask).sum(dims) + 1e-6) / (probability.sum(dims) + mask.sum(dims) + 1e-6)
    return F.binary_cross_entropy_with_logits(logits, mask) + 1 - dice.mean()


def distillation_loss(logits, ema_probability, med_logits, valid, filtered=True, temperature=2.):
    med_probability = med_logits.sigmoid().detach()
    selected = valid[:, None, None, None].expand_as(logits)
    if filtered:
        confident = torch.maximum(med_probability, 1 - med_probability) >= .8
        ema_confident = torch.maximum(ema_probability, 1 - ema_probability) >= .8
        agreement = (ema_probability >= .5) == (med_probability >= .5)
        selected = selected & confident & ema_confident & agreement
    # Bernoulli KL with temperature; detached targets, defined zero for empty selections.
    target = (med_logits.detach() / temperature).sigmoid().clamp(1e-6, 1 - 1e-6)
    student_logp = F.logsigmoid(logits / temperature)
    student_logq = F.logsigmoid(-logits / temperature)
    divergence = target * (target.log() - student_logp) + (1 - target) * ((1 - target).log() - student_logq)
    loss = (divergence * selected).sum() / selected.sum().clamp_min(1) * temperature ** 2
    return loss, selected.float().mean()


def segmentation_metrics(prediction, target):
    prediction, target = prediction.bool(), target.bool()
    tp = (prediction & target).sum().item()
    p, g = prediction.sum().item(), target.sum().item()
    return dict(dice=2 * tp / (p + g) if p + g else 1.,
                precision=tp / p if p else (1. if not g else 0.),
                recall=tp / g if g else (1. if not p else 0.),
                ravd=abs(p - g) / g if g else None,
                relative_volume_error=(p - g) / g if g else None)
