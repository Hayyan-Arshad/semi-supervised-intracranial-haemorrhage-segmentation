import argparse
import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Slices, read_manifest, synthetic
from .learning import distillation_loss, ramp, segmentation_metrics, supervised_loss, update_ema
from .models import Baseline, SAMSegmenter, SmokeNet, boxes_from_probability, full_boxes


MODES = ("supervised", "mean_teacher", "distill", "combined", "filtered")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    patients = {}
    for image, mask, ids in loader:
        image = image.to(device)
        # No ground-truth boxes at evaluation: bootstrap from whole-image prediction.
        initial = model(image, full_boxes(image)).sigmoid()
        boxes, _ = boxes_from_probability(initial)
        prediction = model(image, boxes).sigmoid().cpu() >= .5
        for index, pid in enumerate(ids):
            patients.setdefault(pid, [[], []])
            patients[pid][0].append(prediction[index].flatten())
            patients[pid][1].append(mask[index].flatten())
    per_patient = {pid: segmentation_metrics(torch.cat(p), torch.cat(g)) for pid, (p, g) in patients.items()}
    return {"mean_patient_dice": float(np.mean([m["dice"] for m in per_patient.values()])), "patients": per_patient}


def run(args):
    if args.steps < 1 or args.batch_size < 1 or args.validate_every < 1:
        raise ValueError("steps, batch-size and validate-every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = synthetic(output / "synthetic", args.size) if args.smoke else Path(args.manifest)
    rows = read_manifest(manifest)
    device = torch.device(args.device)
    uses_ema = args.mode in {"mean_teacher", "combined", "filtered"}
    uses_kd = args.mode in {"distill", "combined", "filtered"}
    if not args.smoke and args.backbone == "tiny" and not args.tiny_checkpoint:
        raise ValueError("TinySAM checkpoint required")
    if not args.smoke and uses_kd and not args.medsam_checkpoint:
        raise ValueError("MedSAM checkpoint required for distillation")
    if not args.smoke and uses_kd and args.backbone != "tiny":
        raise ValueError("SAM distillation experiments use the TinySAM student")
    student = (SmokeNet() if args.smoke else SAMSegmenter("tiny", args.tiny_checkpoint, args.vendor, args.adaptation)
               if args.backbone == "tiny" else Baseline(args.backbone)).to(device)
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    med = None
    if uses_kd:
        med = (SmokeNet() if args.smoke else SAMSegmenter("med", args.medsam_checkpoint, args.vendor, args.adaptation)).to(device)
        med.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    loader = DataLoader(Slices(rows, "train_labeled"), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    unlabeled = DataLoader(Slices(rows, "train_unlabeled"), batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed + 1)) if uses_ema or uses_kd else None
    val = DataLoader(Slices(rows, "val"), batch_size=1)
    def endless(source):
        while True:
            yield from source
    labeled_iterator = endless(loader)
    unlabeled_iterator = endless(unlabeled) if unlabeled else None
    metadata = dict(vars(args), manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                    status="synthetic pipeline check; NOT medical performance" if args.smoke else "new experiment; not historical reproduction",
                    torch_version=torch.__version__)
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    best = -1
    ramp_function = ramp
    if not args.smoke:
        source = Path(args.vendor) / "SSL4MIS/code/utils/ramps.py"
        if not source.is_file():
            raise ValueError("Run scripts/fetch_upstream.py to provide SSL4MIS ramp utilities")
        specification = importlib.util.spec_from_file_location("ssl4mis_ramps", source)
        upstream = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(upstream)
        ramp_function = upstream.sigmoid_rampup
    with (output / "history.jsonl").open("w") as history:
        for step in range(args.steps):
            student.train()
            image, mask, _ = next(labeled_iterator)
            image, mask = image.to(device), mask.to(device)
            boxes, _ = boxes_from_probability(mask)
            supervised = supervised_loss(student(image, boxes), mask)
            consistency = supervised.new_zeros(())
            kd, coverage = supervised.new_zeros(()), supervised.new_zeros(())
            if unlabeled_iterator:
                unlab, _, _ = next(unlabeled_iterator)
                unlab = unlab.to(device)
                with torch.no_grad():
                    proposal = ema(unlab, full_boxes(unlab)).sigmoid()
                    prompts, valid = boxes_from_probability(proposal)
                    weak = (unlab + .03 * torch.randn_like(unlab)).clamp(0, 1)
                    target = ema(weak, prompts).sigmoid()
                strong = (unlab + .08 * torch.randn_like(unlab)).clamp(0, 1)
                logits = student(strong, prompts)
                if uses_ema:
                    consistency = (logits.sigmoid() - target).square().mean()
                if uses_kd:
                    with torch.no_grad():
                        med_logits = med(unlab, prompts)
                    kd, coverage = distillation_loss(logits, target, med_logits, valid,
                                                      filtered=args.mode == "filtered")
            weight = ramp_function(step, args.ramp_steps)
            loss = supervised + weight * (args.consistency_weight * consistency + args.kd_weight * kd)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # EMA is also the box proposer in distillation-only mode; no consistency loss there.
            update_ema(student, ema, step)
            record = dict(step=step + 1, loss=loss.item(), supervised=supervised.item(),
                          consistency=consistency.item(), kd=kd.item(), coverage=coverage.item())
            history.write(json.dumps(record) + "\n")
            if (step + 1) % args.validate_every == 0 or step + 1 == args.steps:
                metrics = evaluate(student, val, device)
                if metrics["mean_patient_dice"] > best:
                    best = metrics["mean_patient_dice"]
                    torch.save({"student": student.state_dict(), "ema": ema.state_dict(),
                                "optimizer": optimizer.state_dict(), "step": step + 1, "config": vars(args)}, output / "best.pt")
                    (output / "validation.json").write_text(json.dumps(metrics, indent=2))
                print(json.dumps(dict(record, validation_dice=metrics["mean_patient_dice"])))
    state = torch.load(output / "best.pt", map_location=device, weights_only=True)
    student.load_state_dict(state["student"])
    metrics = evaluate(student, DataLoader(Slices(rows, "test"), batch_size=1), device)
    metrics["status"] = metadata["status"]
    (output / "test.json").write_text(json.dumps(metrics, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--output", default="runs/experiment")
    p.add_argument("--mode", choices=MODES, default="filtered")
    p.add_argument("--backbone", choices=("tiny", "efficientnet", "resnet"), default="tiny")
    p.add_argument("--vendor", default="vendor")
    p.add_argument("--tiny-checkpoint")
    p.add_argument("--medsam-checkpoint")
    p.add_argument("--adaptation", choices=("native", "512"), default="native")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ramp-steps", type=int, default=1000)
    p.add_argument("--consistency-weight", type=float, default=1.)
    p.add_argument("--kd-weight", type=float, default=1.)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--size", type=int, default=64)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
