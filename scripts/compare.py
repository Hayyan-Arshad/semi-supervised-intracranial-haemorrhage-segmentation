"""Render matched CT/reference/prediction panels from trained run directories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ich.data import Slices, read_manifest
from ich.learning import segmentation_metrics
from ich.models import Baseline, SAMSegmenter, SmokeNet, boxes_from_probability, full_boxes

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--output", default="runs/comparison")
p.add_argument("--limit", type=int, default=10)
p.add_argument("--device", default="cpu")
args = p.parse_args()
data = Slices(read_manifest(args.manifest), "test")
models = []
for folder in map(Path, args.runs):
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = (SmokeNet() if config["smoke"] else SAMSegmenter("tiny", config["tiny_checkpoint"], config["vendor"], config["adaptation"])
             if config["backbone"] == "tiny" else Baseline(config["backbone"]))
    model.load_state_dict(checkpoint["student"], strict=True)
    models.append((folder.name, model.to(args.device).eval()))
destination = Path(args.output)
destination.mkdir(parents=True, exist_ok=True)
for index in range(min(args.limit, len(data))):
    image, mask, _ = data[index]
    inputs = image[None].to(args.device)
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4 * (len(models) + 1), 4), squeeze=False, layout="constrained")
    axes = axes[0]
    axes[0].imshow(image[0], cmap="gray", vmin=0, vmax=1)
    if mask.any():
        axes[0].contour(mask[0], levels=[.5], colors=["red"])
    axes[0].set_title("Input / reference (red)")
    for axis, (name, model) in zip(axes[1:], models):
        with torch.no_grad():
            boxes, _ = boxes_from_probability(model(inputs, full_boxes(inputs)).sigmoid())
            prediction = (model(inputs, boxes).sigmoid().cpu()[0] >= .5)
        dice = segmentation_metrics(prediction, mask)["dice"]
        axis.imshow(image[0], cmap="gray", vmin=0, vmax=1)
        if mask.any():
            axis.contour(mask[0], levels=[.5], colors=["red"])
        if prediction.any():
            axis.contour(prediction[0], levels=[.5], colors=["white"])
        axis.set_title(f"{name}\nDice {dice:.3f}")
    for axis in axes:
        axis.axis("off")
    fig.savefig(destination / f"slice_{index:04d}.png", dpi=130)
    plt.close(fig)
print(f"Saved {min(args.limit, len(data))} panels to {destination}; check data permissions before sharing.")
