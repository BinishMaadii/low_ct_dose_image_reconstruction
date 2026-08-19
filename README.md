Low-Dose CT Denoising with RED-CNNAn end-to-end Python pipeline for downloading, parsing, processing, and training a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN) to denoise Low-Dose CT (LDCT) scans using data from The Cancer Imaging Archive (TCIA).Project OverviewData Source: TCIA LDCT-and-Projection-Data Collection (Chest Dataset).Architecture: 10-layer RED-CNN featuring residual connections between encoding and decoding layers.Storage Optimization: Pre-queries TCIA metadata to skip full-manifest downloads and avoid exceeding local working directory quotas.Pipeline Architecture & Workflow ┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
 │ TCIA Manifest  │ ──> │ Metadata Query │ ──> │ Selective DCM  │ ──> │ Export to .npy │
 │  (.tcia file)  │     │   & Matching   │     │ Download (NBIA)│     │ Slices/Patches │
 └────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘
                                                                               │
 ┌────────────────┐     ┌────────────────┐     ┌────────────────┐              │
 │  Visual Plot   │ <── │ PSNR/SSIM Eval │ <── │ Train RED-CNN  │ <───────────┘
 │ (Saved Output) │     │  & Checkpoint  │     │   (PyTorch)    │
 └────────────────┘     └────────────────┘     └────────────────┘
Metadata Querying: Reads a .tcia manifest, queries series-level metadata, strips raw projection data (DICOM-CT-PD), and pairs matching Low-Dose and Full-Dose series by PatientID.Targeted Download: Downloads only the required DICOM series using tcia_utils.nbia.Data Preprocessing: Rescales pixel values to Hounsfield Units (HU) using slope/intercept tags and extracts synchronized slice pairs to normalized .npy arrays.Model Training: Randomly extracts $45 \times 45$ image patches for training a 10-layer RED-CNN using Adam optimizer and MSE loss.Evaluation: Evaluates model performance using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM), saving visual comparisons.Directory Structure.
├── script.py                 # Main pipeline script
├── README.md                 # Project documentation
└── outputs/
    ├── recon/                # Extracted paired numpy slices (.npy)
    │   ├── full_dose/
    │   └── low_dose/
    ├── model/                # Saved PyTorch checkpoints (best.pt)
    └── tcia_pipeline_result.png # Visual evaluation output
Requirements & DependenciesEnsure Python 3.8+ is installed along with the required libraries:Bashpip install torch numpy pandas pydicom tcia_utils scikit-image matplotlib tqdm
Configuration ParametersKey execution variables can be adjusted directly at the top of the script:ParameterDefaultDescriptionMANIFEST_PATH.../Chest.tciaPath to the TCIA manifest file.PATIENT_LIMIT5Maximum number of patient cases to process.DRY_RUNFalseSet True to inspect total download size without fetching images.PATCH_SIZE45Crop dimensions for local training patches ($45 \times 45$).BATCH_SIZE16DataLoader batch size.N_EPOCHS5Total training epochs.UsagePerform a Dry Run (Recommended)Set DRY_RUN = True in the configuration block and execute the script to print metadata reports and verify total memory consumption against working quota limits.Execute Full PipelineSet DRY_RUN = False and run:Bashpython script.py
LicenseThe dataset is governed by the CC BY 4.0 license as specified on The Cancer Imaging Archive.
