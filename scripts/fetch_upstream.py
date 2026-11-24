"""Fetch pinned upstream sources; never fetch patient data or weights."""
import subprocess
from pathlib import Path

SOURCES = {
    "SSL4MIS": ("https://github.com/HiLab-git/SSL4MIS.git", "06df6047a59aba9988ced8331998b5957ecb356b"),
    "TinySAM": ("https://github.com/xinghaochen/TinySAM.git", "11589bc1d98c16cff046c31d5ad4cd90a30f0897"),
}

if __name__ == "__main__":
    for name, (url, revision) in SOURCES.items():
        destination = Path("vendor") / name
        if not destination.exists():
            subprocess.run(["git", "clone", url, str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", revision], check=True)
