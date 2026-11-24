"""Generate README figures from diagrams and synthetic data only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

Path("docs/assets").mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
ax.set(xlim=(0, 12), ylim=(0, 5))
ax.axis("off")
def box(x, y, text, color):
    ax.add_patch(Rectangle((x, y), 2.6, .85, facecolor=color, edgecolor="#333333"))
    ax.text(x + 1.3, y + .425, text, ha="center", va="center", fontsize=10)
def arrow(start, end, label=""):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=15, color="#333333"))
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.12, label, fontsize=9, ha="center")
box(.2, 2, "CT slices\nlabeled + unlabeled", "#e9ecef")
box(4.1, 3.7, "TinySAM student\ntrainable", "#bce4df")
box(4.1, 2, "EMA teacher\nno gradient", "#d8e8ba")
box(4.1, .3, "MedSAM teacher\nfrozen", "#f4d0ca")
box(8.5, 2, "Supervised + consistency\n+ distillation losses", "#e7def1")
arrow((2.8, 2.45), (4.1, 4.1))
arrow((2.8, 2.45), (4.1, 2.45))
arrow((2.8, 2.45), (4.1, .7))
arrow((6.7, 4.1), (8.5, 2.7))
arrow((6.7, 2.45), (8.5, 2.45))
arrow((6.7, .7), (8.5, 2.2))
ax.text(6.1, 4.85, "Reconstructed training pipeline", fontsize=17, ha="center", weight="bold")
ax.text(6, -.02, "Optional distillation filter: teacher confidence and cross-teacher agreement", ha="center", fontsize=10)
fig.savefig("docs/assets/pipeline.png", dpi=160)
plt.close(fig)
rng = np.random.default_rng(17)
y, x = np.mgrid[:128, :128]
mask = ((x-64)**2+(y-64)**2 < 22**2)
image = np.clip(rng.normal(.2, .03, (128, 128)) + .55*mask, 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Synthetic software-test input")
axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[1].contour(mask, levels=[.5], colors=["#ed4539"])
axes[1].set_title("Synthetic reference mask")
for axis in axes:
    axis.axis("off")
fig.suptitle("Not patient images or model predictions", fontsize=13)
fig.savefig("docs/assets/synthetic.png", dpi=160)
