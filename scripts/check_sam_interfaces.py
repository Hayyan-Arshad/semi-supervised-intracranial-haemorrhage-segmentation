"""Random-weight architecture checks; not pretrained medical validation."""
import argparse
import sys
from unittest.mock import patch

import torch

from ich.models import SAMSegmenter, full_boxes

p = argparse.ArgumentParser()
p.add_argument("--vendor", default="vendor")
args = p.parse_args()
sys.path.insert(0, args.vendor + "/TinySAM")
from tinysam import sam_model_registry

torch.set_num_threads(2)
for kind, key in (("tiny", "vit_t"), ("med", "vit_b")):
    model = sam_model_registry[key]()
    state = model.state_dict()
    with patch.dict(sam_model_registry, {key: lambda: model}), patch("torch.load", return_value=state):
        wrapped = SAMSegmenter(kind, "random-weights-interface-check", args.vendor, adaptation="512").eval()
    image = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        result = wrapped(image, full_boxes(image))
    assert result.shape == (1, 1, 512, 512) and torch.isfinite(result).all()
    print(f"{kind}: 512 adaptation forward passed; random weights only", flush=True)
    del wrapped, state, model
