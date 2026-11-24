# Preprocessing

## Implemented input contract

The loader consumes prepared NumPy arrays, not raw DICOM or NIfTI files:

| Array | Shape | Values |
| --- | --- | --- |
| CT input | `3 x H x W` | Finite float32, `[0,1]` |
| Single-window alternative | `H x W` | Repeated into three channels by the loader |
| Reference mask | `H x W` | Binary 0/1; same spatial dimensions as image |

Images are cast to float32. The loader rejects nonfinite values, values outside
`[0,1]`, invalid channel counts and incompatible masks. It does not silently
normalize an incorrectly prepared image. NumPy pickle loading is disabled.

## Preparing CT data externally

Raw-data conversion is dataset-specific and is not implemented in this repository.
For an authorized source, document the following before creating arrays:

1. Convert stored intensities to HU using the source's scaling metadata.
2. Choose and record CT window centers and widths. Use the same channel meaning
   for every patient and source within an experiment.
3. Window in float32 and scale each channel to `[0,1]`.
4. Preserve image-mask alignment, orientation and patient identity.
5. Export binary masks and assign all slices of a patient to the same split.

For window center `c` and width `w > 0`, a reproducible clipping convention is:

```text
low = c - w/2
high = c + w/2
windowed = (clip(HU, low, high) - low) / (high - low)
```

This formula describes an external preparation option, not an automatic loader
operation. No original window settings have been recovered, so none are supplied
as historical defaults. Avoid float16 statistics on raw HU data: cast before
computing any means, standard deviations or clipping operations.

If resizing masks externally, use nearest-neighbor interpolation. Continuous
image channels can use a documented continuous interpolator. The model wrapper
resizes image channels internally using bilinear interpolation and upsamples
logits back to the stored slice dimensions; it does not resize reference masks.
Using a common stored size is necessary for batches larger than one.

## Model-specific normalization

The prepared `[0,1]` image is the common data interface. Inside `SAMSegmenter`:

- TinySAM: multiply by 255, then apply the pretrained SAM pixel mean and standard deviation.
- MedSAM: pass the resized `[0,1]` input directly to the ViT-B encoder.

These paths are not interchangeable. Verify normalization against the exact
pretrained checkpoint release. Patient-wise z-scoring is not performed by the
current loader and z-scored values should not be passed into its `[0,1]` contract.

## Unlabeled data and augmentation

Unlabeled manifest rows must have no mask path. Their returned zero-valued mask
is a batching placeholder and is never used as a negative reference target.
Unlabeled box prompts come from EMA predictions, not hidden annotations.

Training adds weak/strong Gaussian noise after loading and clamps to `[0,1]`.
No elastic deformation, rotations, crop augmentation or boundary-aware loss is
implemented. Those techniques should not be inferred from other project tasks.

## Quality checks

Inspect intensity histograms, channel order, orientation and image-mask overlays
before training. Include positive and negative slices, small lesions and slices
near volume boundaries. Keep source-specific conversion settings with the
dataset record. Do not tune preprocessing on test-set outcomes.

Implementation: [data.py](../ich/data.py), [models.py](../ich/models.py).
