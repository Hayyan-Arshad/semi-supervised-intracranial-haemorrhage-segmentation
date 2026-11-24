# Implementation notes

This is a new reconstruction, not a recovered source tree. Original checkpoints,
patient splits and complete experiment logs are unavailable. No historical
performance results are attributed to this implementation.

The trainer combines TinySAM, a weight-averaged teacher and a frozen MedSAM
teacher. Teacher confidence and agreement filtering is an optional extension.
The supplied tests use synthetic fixtures. Real-data comparisons require
authorized CT data, pretrained checkpoints and matched patient splits.

SSL4MIS supplies the runtime sigmoid ramp-up utility and is the reference for
Mean Teacher training structure, EMA targets and perturbation consistency.
The trainer is independently written. SSL4MIS's dataset-specific slice counts
and CUDA-only entry point are not reused. Its ramps.py carries a CC BY-NC notice;
runtime dependencies retain their original licenses.

TinySAM model definitions are imported from a pinned source checkout. MedSAM
weights must be acquired from its official release. The two 512-input encoder
adaptations preserve the pretrained 64x64 image embedding interface. Decoder
inference uses the single-mask token through predict_masks, because TinySAM's
public forward returns multimasks only.

The repository contains no private messages, patient images or pretrained weights.
