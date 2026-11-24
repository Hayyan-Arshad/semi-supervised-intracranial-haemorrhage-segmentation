import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest is empty")
    required = {"patient_id", "image", "mask", "split"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Required manifest columns: {sorted(required)}")
    patients, images = {}, set()
    for row in rows:
        if not row["patient_id"] or row["split"] not in {"train_labeled", "train_unlabeled", "val", "test"}:
            raise ValueError("Invalid patient ID or split")
        # One patient cannot cross splits, including labeled/unlabeled pools.
        previous = patients.setdefault(row["patient_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Patient leakage: {row['patient_id']}")
        for field in ("image", "mask"):
            if row[field]:
                row[field] = str((path.parent / row[field]).resolve())
                if not Path(row[field]).is_file():
                    raise ValueError(f"Missing {field}: {row[field]}")
        if row["image"] in images:
            raise ValueError("Duplicate image in manifest")
        images.add(row["image"])
        if row["split"] != "train_unlabeled" and not row["mask"]:
            raise ValueError("Labeled/evaluation rows require masks")
        if row["split"] == "train_unlabeled" and row["mask"]:
            raise ValueError("Unlabeled rows must not expose masks")
    return rows


class Slices(Dataset):
    """Pre-windowed CHW float32 CT slices in [0,1]; no implicit rescaling."""

    def __init__(self, rows, split):
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.load(row["image"], allow_pickle=False).astype(np.float32)
        if image.ndim == 2:
            image = np.repeat(image[None], 3, axis=0)
        if image.ndim != 3 or image.shape[0] != 3 or not np.isfinite(image).all():
            raise ValueError("Expected finite CHW image with three channels")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Images must be windowed/scaled to [0,1]")
        mask = np.zeros(image.shape[1:], dtype=np.float32)
        if row["mask"]:
            mask = np.load(row["mask"], allow_pickle=False)
            if mask.shape != image.shape[1:] or not np.isin(mask, [0, 1]).all():
                raise ValueError("Expected binary mask matching image dimensions")
        return torch.from_numpy(image), torch.from_numpy(mask.astype(np.float32)[None]), row["patient_id"]


def synthetic(destination, size=64):
    """Deterministic non-anatomical circles for software tests, not ICH data."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    yy, xx = np.mgrid[:size, :size]
    rows = []
    for split in ("train_labeled", "train_unlabeled", "val", "test"):
        for patient in range(2):
            pid = f"synthetic_{split}_{patient}"
            for index in range(2):
                mask = ((xx - size * .5 - index) ** 2 + (yy - size * .5) ** 2 < (size * .18) ** 2).astype(np.float32)
                image = np.clip(rng.normal(.2, .03, (3, size, size)) + mask * .55, 0, 1).astype(np.float32)
                name = f"{pid}_{index}"
                np.save(destination / f"{name}.npy", image)
                mask_file = ""
                if split != "train_unlabeled":
                    mask_file = f"{name}_mask.npy"
                    np.save(destination / mask_file, mask)
                rows.append(dict(patient_id=pid, image=f"{name}.npy", mask=mask_file, split=split))
    with (destination / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return destination / "manifest.csv"
