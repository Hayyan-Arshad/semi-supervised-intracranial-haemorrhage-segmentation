# Datasets

## Availability

No medical images, annotations, patient splits or pretrained weights are
distributed here. Real experiments require separately acquired, authorized CT
data. The repository accepts prepared NumPy slices through a manifest; it does
not currently provide dataset-specific downloaders or converters.

## Dataset sources

| Resource | Source links | Annotation type |
| --- | --- | --- |
| Subarachnoid haemorrhage CT dataset, Cepeda and Garcia-Garcia | [Zenodo release](https://zenodo.org/records/8228847), [authors' code](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction) | Non-contrast CT volumes and expert manual haemorrhage masks in NIfTI format |
| HemSeg-200 | [Authors' repository](https://github.com/songchangwei/3DCT-SD-IVH-ICH), [dataset paper](https://arxiv.org/abs/2405.14559) | Voxel-level IPH/IVH annotations derived from RSNA CT scans; the original paper describes 222 CT annotations |
| Seg-CQ500 | [Zenodo release](https://zenodo.org/records/8063221), [dataset paper](https://doi.org/10.3389/fnimg.2023.1157565) | Radiologist-produced 3D haemorrhage masks for 51 CQ500 scans |
| RSNA Intracranial Hemorrhage Detection | [RSNA resource page](https://mira.rsna.org/dataset/1), [Kaggle data access](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data) | DICOM images with image-level haemorrhage/subtype classification labels; these are not voxel masks |

These are source resources, not a fixed assignment of datasets to experimental
splits. Specify each selected cohort and release in the experiment manifest.
Project-specific folder aliases such as `SMCCH-SAH` and `SegCQ` must be matched
to the release identifiers when preparing arrays.

### Access and preparation

- Obtain SAH images and masks from the authors' Zenodo record and follow its terms.
- For HemSeg-200, obtain CT images through RSNA/Kaggle and match the authors'
  annotations to their selection metadata. The authors' repository also includes
  a newer HemSeg-500 release; pin the intended release rather than combining versions.
- Use the Seg-CQ500 release and its accompanying paper to identify scan IDs and
  mask definitions. Check the relationship to the parent CQ500 cohort.
- RSNA access requires accepting the competition rules. Images without voxel
  masks can form an unlabeled segmentation pool if authorized and patient-disjoint;
  this is a supported experimental use, not an automatic dataset assignment.

**Cross-source overlap:** HemSeg-200 is derived from RSNA. Exclude its validation
and test patients from any RSNA unlabeled pool. Likewise, do not treat Seg-CQ500
as independent of its parent CQ500 scans. Different directory prefixes do not
make duplicate patients independent.

No image archive is fetched by the setup scripts. Dataset-specific access terms
and conversion remain separate from software installation.

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
