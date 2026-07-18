# AlzheimersFLAIR

This repository contains code for an Alzheimer’s disease MRI analysis project using ADNI FLAIR imaging data. The project explores preprocessing, segmentation, and deep-learning approaches for classifying Alzheimer’s-related diagnostic groups from neuroimaging data.

## Project overview

The goal of this project was to build an end-to-end medical-imaging machine learning workflow, including:

- DICOM to NIfTI conversion
- Brain extraction and preprocessing
- Bias-field correction
- Spatial normalization to MNI152 space
- 2D slice-based model experiments
- 3D CNN / MONAI-based model experiments
- Lateral ventricle segmentation experiments
- Model evaluation using aggregate classification metrics

## Project context

This project was originally developed in 2024 for the Massachusetts Science & Engineering Fair. The public GitHub version was created later after removing restricted ADNI participant-level data, model artifacts, and notebook outputs.

## Data availability

Raw and processed ADNI participant-level data are not included in this repository due to ADNI data-use restrictions. Approved users should obtain data directly through the official ADNI/LONI access process.

This repository includes code, documentation, and project structure only. Participant-level images, labels, processed arrays, derived images, model checkpoints, and subject-specific outputs are intentionally excluded.

## Repository contents

- `data-preprocessing/` — preprocessing notebooks for DICOM conversion, brain extraction, N4 bias correction, and normalization
- `2D-models/` — 2D CNN, transfer learning, and vision transformer experiments
- `3D-models/` — 3D CNN and MONAI-based model experiments
- `LateralVentricleSegmentation/` — U-Net segmentation workflow
- `README.md` — project description and data-use notes

## Methods

The project tested multiple imaging-modeling approaches, including:

- 2D CNNs trained on extracted MRI slices
- Transfer-learning models such as VGG, ResNet, DenseNet, ConvNeXt, and Inception-style architectures
- 3D CNN architectures for volumetric MRI inputs
- MONAI/PyTorch-based medical-imaging workflows
- U-Net-based lateral ventricle segmentation

All notebooks have been cleaned of outputs before public release.

## Results

Aggregate model results and parameter summaries are included where they do not contain participant-level information. Raw predictions, subject-level labels, medical images, processed arrays, and trained model weights are excluded.

## Reproducibility

To reproduce the workflow, approved ADNI users should:

1. Obtain ADNI data through the official ADNI/LONI access process.
2. Place raw data locally according to the folder structure expected by the notebooks.
3. Run the preprocessing notebooks in `data-preprocessing/`.
4. Train/evaluate models using the notebooks in `2D-models/`, `3D-models/`, or `LateralVentricleSegmentation/`.

Because ADNI data are restricted, this repository is intended to demonstrate the codebase and methodology rather than provide a fully self-contained runnable dataset.

## Disclaimer

This project is for educational and research purposes only. It is not a clinical diagnostic tool.
