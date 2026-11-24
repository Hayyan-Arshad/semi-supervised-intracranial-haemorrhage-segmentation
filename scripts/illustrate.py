"""Generate README figures from diagrams and synthetic data only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

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
