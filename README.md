# Low-Dose CT Denoising with RED-CNN

An end-to-end Python pipeline for downloading, parsing, processing, and training a **Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)** to denoise Low-Dose CT (LDCT) scans using data from **The Cancer Imaging Archive (TCIA)**.

## Project Overview

* **Data Source:** [TCIA LDCT-and-Projection-Data Collection](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) (Chest Dataset)
* **Architecture:** 10-layer RED-CNN featuring residual connections between encoding and decoding layers
* **Storage Optimization:** Pre-queries TCIA metadata to skip full-manifest downloads and avoid exceeding local working directory quotas

---

## Workflow

1. **Metadata Querying:** Reads a `.tcia` manifest, queries series-level metadata, strips raw projection data (`DICOM-CT-PD`), and pairs matching Low-Dose and Full-Dose series by `PatientID`.
2. **Targeted Download:** Downloads *only* the required DICOM series using `tcia_utils.nbia`.
3. **Data Preprocessing:** Rescales pixel values to Hounsfield Units (HU) using slope/intercept tags and extracts synchronized slice pairs to normalized `.npy` arrays.
4. **Model Training:** Randomly extracts $45 \times 45$ image patches for training a 10-layer RED-CNN using Adam optimizer and MSE loss.
5. **Evaluation:** Evaluates model performance using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM), saving visual comparisons.

---

## Requirements & Dependencies
Ensure Python 3.8+ is installed along with the required libraries:

$Bash

$pip install torch numpy pandas pydicom tcia_utils scikit-image matplotlib tqdm


## Configuration Parameters

Key execution variables can be adjusted directly at the top of the script:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MANIFEST_PATH` | `.../Chest.tcia` | Path to the TCIA manifest file |
| `PATIENT_LIMIT` | `5` | Maximum number of patient cases to process |
| `DRY_RUN` | `False` | Set `True` to inspect total download size without fetching images |
| `PATCH_SIZE` | `45` | Crop dimensions for local training patches |
| `BATCH_SIZE` | `16` | DataLoader batch size |
| `N_EPOCHS` | `5` | Total training epochs |

---

## Usage

### 1. Perform a Dry Run (Recommended)
Set `DRY_RUN = True` in the configuration block and execute the script to print metadata reports and verify total memory consumption against working quota limits.

### 2. Execute Full Pipeline
Set `DRY_RUN = False` and run:

```bash
python script.py

