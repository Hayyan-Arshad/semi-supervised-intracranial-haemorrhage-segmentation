"""Run paired ablations. Synthetic smoke results are not medical evidence."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--output", default="runs/ablations")
args, training_options = parser.parse_known_args()
for seed in args.seeds:
    for mode in ("supervised", "mean_teacher", "distill", "combined", "filtered"):
        output = Path(args.output) / f"{mode}_seed{seed}"
        command = [sys.executable, "-m", "ich.train", "--mode", mode, "--seed", str(seed),
                   "--steps", str(args.steps), "--output", str(output)]
        if args.smoke:
            command.append("--smoke")
        subprocess.run(command + training_options, check=True)
summary = []
for path in sorted(Path(args.output).glob("*/test.json")):
    summary.append({"run": path.parent.name, **json.loads(path.read_text())})
(Path(args.output) / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"Saved {len(summary)} runs; inspect status labels before interpreting scores.")
