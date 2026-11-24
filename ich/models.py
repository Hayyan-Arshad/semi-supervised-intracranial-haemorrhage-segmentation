import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SmokeNet(nn.Module):
    """Small substitute used ONLY to verify pipeline execution."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, image, boxes=None):
        return self.net(image)


class SAMSegmenter(nn.Module):
    """One box per slice; loop over samples to respect SAM decoder batching."""

    def __init__(self, kind, checkpoint, vendor, adaptation="native"):
        super().__init__()
        if adaptation not in {"native", "512"}:
            raise ValueError("Unknown encoder adaptation")
        root = Path(vendor).resolve()
        if not (root / "TinySAM" / "tinysam").is_dir():
            raise ValueError("Run scripts/fetch_upstream.py first")
        sys.path.insert(0, str(root / "TinySAM"))
        from tinysam import sam_model_registry
        self.sam = sam_model_registry["vit_t" if kind == "tiny" else "vit_b"]()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.sam.load_state_dict(state, strict=True)
        self.kind, self.adaptation = kind, adaptation
        self.coordinate_size = 1024
        if adaptation == "512":
            if kind == "tiny":
                # Preserve the 256x256 patch grid expected by pretrained TinyViT.
                self.sam.image_encoder.patch_embed.seq[0].c.stride = (1, 1)
            else:
                # ViT-B patch embeddings: 32x32 -> 64x64 before positional encoding.
                self.sam.image_encoder.patch_embed.register_forward_hook(
                    lambda module, inputs, output: F.interpolate(
                        output.permute(0, 3, 1, 2), size=(64, 64), mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1)
                )
        for parameter in self.sam.prompt_encoder.parameters():
            parameter.requires_grad_(False)

    def forward(self, image, boxes):
        original_size = image.shape[-2:]
        encoder_size = 512 if self.adaptation == "512" else 1024
        image = F.interpolate(image, (encoder_size, encoder_size), mode="bilinear", align_corners=False)
        # MedSAM consumes medical images scaled to [0,1]; TinySAM uses SAM statistics.
        if self.kind == "tiny":
            image = (image * 255 - self.sam.pixel_mean) / self.sam.pixel_std
        embeddings = self.sam.image_encoder(image)
        scale = image.new_tensor([1024 / original_size[1], 1024 / original_size[0]] * 2)
        output = []
        for embedding, box in zip(embeddings, boxes * scale):
            with torch.no_grad():
                sparse, dense = self.sam.prompt_encoder(points=None, boxes=box[None], masks=None)
            # TinySAM's public forward returns multimasks only; select the
            # pretrained single-mask token through its shared decoder API.
            logits, _ = self.sam.mask_decoder.predict_masks(
                image_embeddings=embedding[None], image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            )
            output.append(F.interpolate(logits[:, :1], original_size, mode="bilinear", align_corners=False))
        return torch.cat(output)


class Baseline(nn.Module):
    def __init__(self, kind):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder_name="efficientnet-b2" if kind == "efficientnet" else "resnet18",
                            encoder_weights=None, in_channels=3, classes=1)

    def forward(self, image, boxes=None):
        return self.net(image)


@torch.no_grad()
def boxes_from_probability(probability, threshold=.5):
    boxes, valid = [], []
    height, width = probability.shape[-2:]
    for mask in probability[:, 0] >= threshold:
        coordinates = mask.nonzero()
        valid.append(bool(coordinates.numel()))
        if coordinates.numel():
            lo, hi = coordinates.amin(0), coordinates.amax(0)
            boxes.append(torch.stack((lo[1], lo[0], hi[1] + 1, hi[0] + 1)))
        else:
            boxes.append(probability.new_tensor([0, 0, width, height]))
    return torch.stack(boxes).to(probability), torch.tensor(valid, device=probability.device)


def full_boxes(image):
    return image.new_tensor([0, 0, image.shape[-1], image.shape[-2]]).repeat(image.shape[0], 1)
