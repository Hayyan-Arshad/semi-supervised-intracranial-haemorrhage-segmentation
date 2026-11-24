# Datasets

## Availability

No medical images, annotations, original splits or pretrained weights are
distributed here. Real experiments require separately acquired, authorized CT
data. The repository accepts prepared NumPy slices through a manifest; it does
not currently provide dataset-specific downloaders or converters.

The earlier project context included the names SMCCH-SAH, Hem-Seg-200, RSNA and
SegCQ. Exact versions, access terms, sample counts and final labeled/unlabeled
assignments have not been recovered. These names are provenance leads, not a
claim that four datasets are integrated or benchmarked in this repository.
In particular, RSNA's role as an unlabeled source is not established.

Do not infer pixel-level annotations from classification labels. A slice-level
hemorrhage label alone cannot supervise a segmentation mask. Identify actual
annotation types and eligibility separately for each source.

## Manifest format

```csv
patient_id,image,mask,split
sourceA_p001,images/p001_001.npy,masks/p001_001.npy,train_labeled
sourceA_p002,images/p002_001.npy,,train_unlabeled
sourceA_p003,images/p003_001.npy,masks/p003_001.npy,val
sourceA_p004,images/p004_001.npy,masks/p004_001.npy,test
```

Relative paths resolve against the manifest directory; absolute paths are also
accepted. Each row is a slice. Multiple rows may share a patient identifier if
they have the same split. Prefix identifiers with a source name when necessary
to avoid accidental collisions, while preserving real cross-source duplicates.

| Split | Mask required | Used for |
| --- | --- | --- |
| `train_labeled` | yes | Supervised loss and reference-derived training boxes |
| `train_unlabeled` | forbidden | EMA consistency and MedSAM distillation |
| `val` | yes | Student checkpoint selection |
| `test` | yes | Final selected-checkpoint evaluation |

The loader rejects missing files, duplicate resolved image paths, cross-split
patient IDs and mask paths in unlabeled rows. It cannot detect duplicate scans
stored in different files, renamed patients, overlapping cohorts or identity
collisions concealed by different IDs. Check these before making a manifest.

## Patient-level splitting

Construct the patient partition before creating slice manifests. Keep all scans
and slices from an individual in one split, including labeled versus unlabeled
training pools under the current validator. Reuse the exact manifest for every
ablation. Matching random seeds without matching patient identities is not
sufficient to prevent leakage or guarantee a comparable test set.

The repository does not generate train/validation/test ratios or patient-wise
cross-validation folds. If using multiple folds, prepare an explicit manifest
per fold and report the fold-construction method separately.

## Record before reporting results

- Dataset version, access source and permitted uses.
- Patient and slice counts per split, including negative slices.
- Annotation definitions, subtype-to-binary mapping and quality controls.
- HU conversion, window parameters, image orientation and stored slice size.
- Patient deduplication and partition creation procedure.
- Physical spacing and whether all eligible slices are included for volume metrics.
- Manifest and data-content hashes, plus any exclusions decided before evaluation.

Keep identifiable patient IDs, private paths and generated patient images out
of public commits. `data/`, `checkpoints/` and `runs/` are ignored by Git, but
ignore rules are not a substitute for checking staged files.

## Synthetic fixtures

The software tests generate small non-anatomical shapes with patient-like IDs
and disjoint splits. They verify data handling and training execution only.
They are not a substitute for any clinical dataset or medical-data ablation.

Related: [preprocessing](preprocessing.md), [training details](training_details.md).
