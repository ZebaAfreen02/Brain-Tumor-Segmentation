# Brain Tumor Segmentation Using Multi-Modal MRI and U-Net

**Deep Learning | PyTorch | Computer Vision | Medical Image Segmentation**

A deep learning pipeline for multi-class brain tumor segmentation using a U-Net architecture trained on multi-modal MRI scans from the BraTS 2020 dataset.

The model combines four MRI modalities—T1, T1CE, T2, and FLAIR—to identify tumor sub-regions at the pixel level.

## Key Results

| Metric | Reported result |
|---|---|
| Training Dice score | ~0.91 |
| Validation Dice score | ~0.88 |
| Final training loss (saved notebook, epoch 40) | 0.1095 |
| Final validation loss (saved notebook, epoch 40) | 0.1561 |
| Training epochs | 40 |
| Input dimensions | 240 × 240 × 4 |

The model demonstrated close agreement between predicted segmentation masks and ground-truth annotations in the qualitative examples.

The epoch-40 losses are from the saved notebook output and have not been independently reproduced.

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

The checked-in notebook splits patient directories before individual slices enter the data loaders. After excluding case 355, it holds out 15% for testing and then 20% of the remaining patients for validation, using random seed `42` in both steps.

| Split | Patients | Share |
|---|---:|---:|
| Training | 249 | 67.7% |
| Validation | 63 | 17.1% |
| Test | 56 | 15.2% |
| Total | 368 | 100% |

These counts come from the saved notebook output. The original academic report describes a 70% / 20% / 10% split.

If a segmentation mask needs resizing, the notebook uses OpenCV's default interpolation, which can introduce intermediate values between discrete class labels. The saved results do not establish whether this affected the original run.

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
| Optimizer weight decay | None (`weight_decay=0`) |
| Training data augmentation | Random horizontal flips, random rotations of 0°, 90°, 180°, or 270°, and occasional brightness scaling |

The same spatial transformations are applied to each image and its mask. Brightness scaling applies only to the image. Validation uses no augmentation.

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

Use Python 3.10. Running the checked-in notebook requires a CUDA-enabled GPU and a CUDA-enabled PyTorch build because the notebook calls `.cuda()` directly.

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
python -m pip install numpy nibabel opencv-python matplotlib tqdm torchinfo jupyter
```

### 4. Run the notebook

Open `Brain Tumor Segmentation/final-project.ipynb` in Jupyter Notebook or VS Code.

The BraTS 2020 dataset is not included. In notebook cell 3, set `TRAIN_PATH` to the extracted BraTS 2020 training directory containing the patient folders. The notebook derives its validation and test sets from this directory. Set `OUTPUT_PATH` if you want to store or reuse preprocessed arrays in a location other than `./preprocessed`.

On a first run, or if the preprocessed arrays are missing, uncomment and execute the preprocessing loop in cell 6 after running cells 1–4. It writes per-patient `<case_id>_images.npy` and `<case_id>_masks.npy` files under `OUTPUT_PATH`. The data loaders and prediction cells read those arrays.

The training cell writes `best_model.pth` when validation Dice improves. The following save cell writes `history.pkl` and the final checkpoint `brats_unet_model.pth`. The later reload and evaluation cells expect `best_model.pth` and `history.pkl` in the notebook's working directory. The repository includes the notebook and academic report. The arrays, checkpoints, and history file are generated locally.

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
