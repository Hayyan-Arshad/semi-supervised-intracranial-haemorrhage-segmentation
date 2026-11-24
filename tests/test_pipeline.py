import copy
import csv

import numpy as np
import pytest
import torch

from ich.data import Slices, read_manifest, synthetic
from ich.learning import distillation_loss, ramp, segmentation_metrics, update_ema
from ich.models import SmokeNet, boxes_from_probability
from ich.train import parser, run


def test_patient_leakage(tmp_path):
    manifest = synthetic(tmp_path)
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["patient_id"] = rows[0]["patient_id"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(manifest)


def test_unlabeled_has_no_annotations(tmp_path):
    rows = read_manifest(synthetic(tmp_path))
    _, mask, _ = Slices(rows, "train_unlabeled")[0]
    assert mask.sum() == 0


def test_boxes_empty_and_nonempty():
    probability = torch.zeros(2, 1, 8, 10)
    probability[1, 0, 2:4, 3:6] = 1
    boxes, valid = boxes_from_probability(probability)
    assert boxes.tolist() == [[0, 0, 10, 8], [3, 2, 6, 4]]
    assert valid.tolist() == [False, True]


def test_filter_empty_differentiable_and_no_teacher_grad():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    med = torch.ones_like(logits, requires_grad=True)
    loss, coverage = distillation_loss(logits, logits.detach().sigmoid(), med, torch.ones(2, dtype=torch.bool))
    loss.backward()
    assert loss == 0 and coverage == 0
    assert med.grad is None and torch.isfinite(logits.grad).all()


def test_kd_selected_updates_student():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss, coverage = distillation_loss(logits, torch.ones_like(logits)*.99, torch.ones_like(logits)*5, torch.tensor([True]))
    loss.backward()
    assert coverage == 1 and logits.grad.abs().sum() > 0


def test_ema_and_metrics():
    student = SmokeNet()
    teacher = copy.deepcopy(student).requires_grad_(False)
    with torch.no_grad():
        next(student.parameters()).add_(1)
    update_ema(student, teacher, 0)
    assert torch.equal(next(student.parameters()), next(teacher.parameters()))
    result = segmentation_metrics(torch.tensor([1, 1, 0]), torch.tensor([1, 0, 0]))
    assert result["dice"] == pytest.approx(2/3)
    assert result["ravd"] == 1
    assert ramp(100, 100) == 1


@pytest.mark.parametrize("mode", ["supervised", "mean_teacher", "distill", "combined", "filtered"])
def test_complete_smoke_run(tmp_path, mode):
    args = parser().parse_args(["--smoke", "--mode", mode, "--steps", "2", "--size", "16", "--output", str(tmp_path)])
    run(args)
    assert (tmp_path / "best.pt").is_file()
    assert "NOT medical performance" in (tmp_path / "test.json").read_text()
