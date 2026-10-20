# Brain Tumor Segmentation Using Multi-Modal MRI and U-Net

**Deep Learning | PyTorch | Computer Vision | Medical Image Segmentation**

A deep learning pipeline for multi-class brain tumor segmentation using a U-Net architecture trained on multi-modal MRI scans from the BraTS 2020 dataset.

The model combines four MRI modalities—T1, T1CE, T2, and FLAIR—to identify tumor sub-regions at the pixel level.

## Key Results

| Metric | Reported result |
|---|---|
| Training Dice score | ~0.91 |
| Validation Dice score | ~0.88 |
| Final training loss | ~0.12–0.16 |
| Training epochs | 40 |
| Input dimensions | 240 × 240 × 4 |

The model demonstrated close agreement between predicted segmentation masks and ground-truth annotations in the qualitative examples.

## Problem Statement

Accurate brain tumor segmentation is important for medical image analysis, treatment planning, and monitoring.

Manual segmentation of MRI scans is time-consuming and can vary between observers. This project explores how deep learning can automate the identification of tumor regions using complementary information from multiple MRI modalities.

The implementation is a research and educational prototype, not a clinically validated diagnostic tool.

## Dataset

**Dataset:** BraTS 2020 (Brain Tumor Segmentation Challenge)

The dataset described in the project report contains 369 patient MRI volumes, with four imaging modalities:

| Modality | Description |
|---|---|
| T1 | Structural brain anatomy |
| T1CE | Contrast-enhanced tumor regions |
| T2 | Fluid-rich regions and edema |
| FLAIR | Tumor boundaries and surrounding edema |

Each 3D MRI volume is processed into 2D slices with four input channels.

### Segmentation Classes

| Class | Region |
|---|---|
| 0 | Background |
| 1 | Necrotic / non-enhancing tumor (NCR/NET) |
| 2 | Edema (ED) |
| 3 | Enhancing tumor (ET) |

## Methodology

### 1. Data Preprocessing

- Load MRI volumes and their corresponding segmentation masks.
- Extract 2D slices from the 3D volumes.
- Normalize MRI intensity values.
- Stack the four MRI modalities into a single input tensor.
- Encode segmentation labels into the model's four target classes.
- Split patient data into training, validation, and test sets.

### 2. U-Net Architecture

The segmentation model uses an encoder–decoder architecture with skip connections.

**Encoder:** Convolutional blocks progressively extract spatial and semantic features while reducing resolution through pooling.

**Bottleneck:** Captures high-level contextual information about the tumor and surrounding tissue.

**Decoder:** Upsamples the learned representations and combines them with encoder features through skip connections to recover spatial detail.

**Output:** Four-channel segmentation predictions representing background and three tumor sub-regions.

### 3. Training

| Configuration | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 16 |
| Epochs | 40 |
| Loss function | Dice loss + cross-entropy loss |
| Regularization | Weight decay |
| Data augmentation | Flips |

The model is evaluated using the Dice similarity coefficient, which measures overlap between predicted and ground-truth segmentation regions.

## Results and Evaluation

The project report describes stable training and validation convergence, with a final validation Dice score of approximately **0.88** and training Dice score of approximately **0.91**.

Qualitative evaluation compares predictions against ground-truth segmentation masks across different MRI slices.

The model successfully identifies the principal tumor sub-regions in the illustrated examples, although these results should not be interpreted as clinical validation.

For detailed training curves, architecture diagrams, and segmentation visualizations, see the [project report](Brain%20Tumor%20Segmentation/ML_Project_Report.pdf).

## Repository Structure

```text
Brain-Tumor-Segmentation/
├── README.md
└── Brain Tumor Segmentation/
    ├── final-project.ipynb
    ├── ML_Project_Report.pdf
    └── readme.md
```

- [Training and evaluation notebook](Brain%20Tumor%20Segmentation/final-project.ipynb)
- [Project report](Brain%20Tumor%20Segmentation/ML_Project_Report.pdf)

## Installation

Python 3.10 and a CUDA-compatible GPU are recommended for the existing notebook.

### 1. Create an environment

```bash
conda create -n brain-tumor-segmentation python=3.10 -y
conda activate brain-tumor-segmentation
```

### 2. Install PyTorch

For a CUDA-compatible system, install a supported PyTorch build using the instructions at:

https://pytorch.org/get-started/locally/

### 3. Install dependencies

```bash
pip install numpy nibabel opencv-python scikit-learn matplotlib tqdm jupyter
```

### 4. Run the notebook

Open `Brain Tumor Segmentation/final-project.ipynb` in Jupyter Notebook or VS Code.

Configure the dataset paths for your local BraTS 2020 download before running the cells.

**Note:** The current implementation contains CUDA-specific calls and requires adjustments to run on CPU-only hardware.

## Future Improvements

- Evaluate performance separately for each tumor sub-region.
- Add additional segmentation metrics and systematic error analysis.
- Improve portability across GPU and CPU environments.
- Refactor notebook code into reusable preprocessing, training, and evaluation modules.
- Explore additional architectures and augmentation strategies.

## Project Context

This project was developed collaboratively as part of graduate-level machine learning coursework.

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI, 2015.
- Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)*, IEEE Transactions on Medical Imaging, 2015.
- Bakas et al., *Advancing the Cancer Genome Atlas Glioma MRI Collections with Expert Segmentations*, Scientific Data, 2017.

For additional references, see the accompanying project report.
