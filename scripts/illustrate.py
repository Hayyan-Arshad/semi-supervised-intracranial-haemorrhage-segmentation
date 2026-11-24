"""Generate technical schematics; no patient images or measured result plots."""
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
ax.text(6.1, 4.85, "Semi-supervised ICH segmentation", fontsize=17, ha="center", weight="bold")
ax.text(6, -.02, "Optional distillation filter: teacher confidence and cross-teacher agreement", ha="center", fontsize=10)
fig.savefig("docs/assets/pipeline.png", dpi=160)
plt.close(fig)


def flow_figure(filename, title, stages, footer):
    fig, axes = plt.subplots(figsize=(14, 3.5), layout="constrained")
    axes.set(xlim=(0, 14), ylim=(0, 3.5))
    axes.axis("off")
    colors = ["#e9ecef", "#d8e8ba", "#bce4df", "#e7def1"]
    for index, label in enumerate(stages):
        x = .25 + index * 3.5
        axes.add_patch(Rectangle((x, 1.1), 2.95, 1.15,
                                 facecolor=colors[index], edgecolor="#333333"))
        axes.text(x + 1.475, 1.675, label, ha="center", va="center", fontsize=11)
        if index < len(stages) - 1:
            axes.add_patch(FancyArrowPatch((x+2.95, 1.675), (x+3.5, 1.675),
                                          arrowstyle="->", mutation_scale=17))
    axes.text(7, 3, title, ha="center", fontsize=17, weight="bold")
    axes.text(7, .35, footer, ha="center", fontsize=10)
    fig.savefig(f"docs/assets/{filename}.png", dpi=160)
    plt.close(fig)


flow_figure("prompt_flow", "Spatial prompts without unlabeled annotations", [
    "Unlabeled CT slice\nwhole-image box",
    "EMA prediction\nforeground threshold 0.5",
    "Enclosing foreground box\nsource-image coordinates",
    "Shared spatial prompt\nTinySAM + MedSAM",
], "Empty foreground: whole-image fallback; exclude the slice from distillation.")

flow_figure("teacher_filter", "Confidence and agreement selection", [
    "EMA + MedSAM\nforeground probabilities",
    "Both confidences >= 0.8\nbinary predictions agree",
    "Valid proposal mask\nintersect selected pixels",
    "Selected-pixel KL loss\ntemperature T = 2",
], "Selection uses untempered probabilities; distillation targets are detached.")

fig, axes = plt.subplots(figsize=(11, 4.5), layout="constrained")
matrix = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
from matplotlib.colors import ListedColormap
axes.imshow(matrix, cmap=ListedColormap(["#eeeeee", "#73b8a9"]), vmin=0, vmax=1, aspect="auto")
axes.set_xticks(range(4), ["Supervised loss", "EMA consistency", "MedSAM distillation", "Confidence / agreement"])
axes.set_yticks(range(5), ["Supervised", "Mean Teacher", "Distillation", "Combined", "Filtered"])
axes.tick_params(length=0, pad=12)
for row in range(5):
    for column in range(4):
        axes.text(column, row, "ON" if matrix[row][column] else "OFF", ha="center", va="center", fontsize=12)
axes.set_title("Ablation design", fontsize=17, weight="bold", pad=20)
axes.set_xlabel("Configuration matrix, not measured performance", labelpad=15)
for spine in axes.spines.values():
    spine.set_visible(False)
fig.savefig("docs/assets/ablations.png", dpi=160)
plt.close(fig)
