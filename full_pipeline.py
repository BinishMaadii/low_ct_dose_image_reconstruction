# ============================================================================
# LOW-DOSE CT DENOISING WITH RED-CNN
# One pipeline, start to end, 5 patients.
# Copy each STEP into its own Kaggle cell, or run the whole file top to bottom.
# ============================================================================


# ############################################################################
# STEP 1: CONFIGURATION
# ############################################################################
MANIFEST_PATH = "/kaggle/input/datasets/binishbatool/low-ct-dataa/LDCT-and-Projection-data_v07_20260520_Chest.tcia"
DOWNLOAD_DIR = "/kaggle/working/tcia_data"
OUTPUT_DIR = "/kaggle/working/outputs"

PATIENT_LIMIT = 5       # number of patients to use
N_EPOCHS = 5            # raise to 20+ for a real run
PATCH_SIZE = 64
BATCH_SIZE = 16
PATCHES_PER_SLICE = 4
LEARNING_RATE = 1e-4
EVAL_EVERY_N_SLICES = 10   # final metrics are computed on every 10th slice (keeps it fast)


# ############################################################################
# STEP 2: IMPORTS AND FOLDERS
# ############################################################################
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tcia_utils", "pydicom"], check=True)

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from IPython.display import Image, FileLink, display

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

manifest_path = Path(MANIFEST_PATH)
download_dir = Path(DOWNLOAD_DIR)
output_dir = Path(OUTPUT_DIR)
recon_dir = output_dir / "recon"
model_dir = output_dir / "model"
plot_dir = output_dir / "plots"

for folder in [output_dir, recon_dir / "low_dose", recon_dir / "full_dose", model_dir, plot_dir]:
    folder.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Device: %s", device)

saved_plots = []   # every plot path is added here, used later for display and links


# ############################################################################
# STEP 3: READ THE MANIFEST AND PAIR FULL-DOSE / LOW-DOSE SERIES
# ############################################################################
from tcia_utils import nbia

FULL_DOSE_HINTS = ("full", "normal", "ndct", "high", "target", "hd")
LOW_DOSE_HINTS = ("low", "quarter", "ldct", "input", "ld")

all_uids = nbia.manifestToList(str(manifest_path))
meta = nbia.getSeriesList(uids=all_uids)

desc_col = "SeriesDescription" if "SeriesDescription" in meta.columns else "Series Description"
patient_col = "PatientID" if "PatientID" in meta.columns else "Patient ID"

# drop the raw projection series, keep only reconstructed images
is_projection = meta[desc_col].astype(str).str.contains("projection", case=False, na=False)
image_meta = meta[~is_projection].copy()

# label every series as "low" or "full"
dose_labels = []
for text in image_meta[desc_col].astype(str):
    t = text.lower()
    if any(h in t for h in LOW_DOSE_HINTS):
        dose_labels.append("low")
    elif any(h in t for h in FULL_DOSE_HINTS):
        dose_labels.append("full")
    else:
        dose_labels.append(None)
image_meta["dose"] = dose_labels
image_meta = image_meta[image_meta["dose"].notna()]

# group the series UIDs by patient
tagged = {}
for _, row in image_meta.iterrows():
    pid = str(row[patient_col])
    tagged.setdefault(pid, {"full": [], "low": []})[row["dose"]].append(str(row["SeriesInstanceUID"]))

# build one pair per patient (full + low)
all_pairs = []
for pid, doses in sorted(tagged.items()):
    for i in range(min(len(doses["full"]), len(doses["low"]))):
        all_pairs.append({"patient_id": pid, "full_uid": doses["full"][i], "low_uid": doses["low"][i]})

pairs = all_pairs[:PATIENT_LIMIT]
patient_ids = [p["patient_id"] for p in pairs]

print(f"Series in manifest: {len(meta)}")
print(f"Image series with a dose label: {len(image_meta)}")
print(f"Patients with a full+low pair: {len(all_pairs)}")
print(f"Using {len(pairs)} patients: {patient_ids}")


# ############################################################################
# STEP 4: DOWNLOAD THE SELECTED SERIES
# ############################################################################
uids_to_download = []
for p in pairs:
    uids_to_download.append(p["full_uid"])
    uids_to_download.append(p["low_uid"])

log.info("Downloading %d series ...", len(uids_to_download))
nbia.downloadSeries(uids_to_download, input_type="list", path=str(download_dir))


# ############################################################################
# STEP 5: CONVERT DICOM TO HU SLICES AND SAVE AS .npy
# ############################################################################
slice_counts = []   # one row per patient, used for plots later

for p in pairs:
    volumes = {}
    for dose in ["low", "full"]:
        series_dir = download_dir / p[dose + "_uid"]
        dcm_files = sorted(series_dir.glob("*.dcm")) or sorted(f for f in series_dir.iterdir() if f.is_file())
        slices = [pydicom.dcmread(f) for f in dcm_files]
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))

        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
        slope = float(getattr(slices[0], "RescaleSlope", 1.0))
        intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
        volumes[dose] = volume * slope + intercept

    n_slices = min(volumes["low"].shape[0], volumes["full"].shape[0])   # keep low/full aligned
    for i in range(n_slices):
        np.save(recon_dir / "low_dose" / f"{p['patient_id']}_{i:04d}.npy", volumes["low"][i])
        np.save(recon_dir / "full_dose" / f"{p['patient_id']}_{i:04d}.npy", volumes["full"][i])

    slice_counts.append({"patient": p["patient_id"], "n_slices": n_slices,
                         "height": volumes["full"].shape[1], "width": volumes["full"].shape[2]})
    log.info("Patient %s: %d slices exported", p["patient_id"], n_slices)

slice_table = pd.DataFrame(slice_counts)
print(slice_table)


# ############################################################################
# STEP 6: DATA ANALYSIS AND DISTRIBUTION PLOTS
# ############################################################################

# --- 6a. example slice (middle slice of each patient): low dose vs full dose ---
fig, axes = plt.subplots(len(patient_ids), 2, figsize=(8, 3.6 * len(patient_ids)))
for row, pid in enumerate(patient_ids):
    mid = int(slice_table.loc[slice_table["patient"] == pid, "n_slices"].iloc[0]) // 2
    low_img = np.load(recon_dir / "low_dose" / f"{pid}_{mid:04d}.npy")
    full_img = np.load(recon_dir / "full_dose" / f"{pid}_{mid:04d}.npy")
    axes[row, 0].imshow(low_img, cmap="gray", vmin=-1000, vmax=400)
    axes[row, 0].set_title(f"{pid}  low dose (slice {mid})", fontsize=10)
    axes[row, 1].imshow(full_img, cmap="gray", vmin=-1000, vmax=400)
    axes[row, 1].set_title(f"{pid}  full dose (slice {mid})", fontsize=10)
    axes[row, 0].axis("off")
    axes[row, 1].axis("off")
plt.tight_layout()
path = plot_dir / "01_example_slices_all_patients.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)

# --- 6b. slices per patient ---
plt.figure(figsize=(7, 4))
plt.bar(slice_table["patient"], slice_table["n_slices"], color="steelblue")
plt.title("Number of paired slices per patient")
plt.xlabel("Patient")
plt.ylabel("Slices")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
path = plot_dir / "02_slices_per_patient.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)

# --- 6c. HU value distribution, low vs full (every 10th slice, all patients) ---
low_values = []
full_values = []
noise_rows = []     # noise = low - full, per patient
noise_samples = []

for pid in patient_ids:
    n = int(slice_table.loc[slice_table["patient"] == pid, "n_slices"].iloc[0])
    patient_noise = []
    for i in range(0, n, 10):
        low_img = np.load(recon_dir / "low_dose" / f"{pid}_{i:04d}.npy")
        full_img = np.load(recon_dir / "full_dose" / f"{pid}_{i:04d}.npy")
        low_values.append(low_img[::4, ::4].ravel())     # subsample pixels to save memory
        full_values.append(full_img[::4, ::4].ravel())
        diff = (low_img - full_img)[::4, ::4].ravel()
        patient_noise.append(diff)
    patient_noise = np.concatenate(patient_noise)
    noise_samples.append(patient_noise)
    noise_rows.append({"patient": pid, "noise_mean_HU": patient_noise.mean(), "noise_std_HU": patient_noise.std()})

low_values = np.concatenate(low_values)
full_values = np.concatenate(full_values)
noise_table = pd.DataFrame(noise_rows)

plt.figure(figsize=(8, 4.5))
plt.hist(full_values, bins=200, alpha=0.6, label="Full dose", color="tab:green", log=True)
plt.hist(low_values, bins=200, alpha=0.6, label="Low dose", color="tab:red", log=True)
plt.title("Pixel value (HU) distribution, all patients")
plt.xlabel("HU")
plt.ylabel("Pixel count (log scale)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
path = plot_dir / "03_hu_distribution_low_vs_full.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)

# --- 6d. noise distribution (low - full) per patient ---
plt.figure(figsize=(8, 4.5))
for pid, noise in zip(patient_ids, noise_samples):
    plt.hist(noise, bins=200, range=(-300, 300), alpha=0.45, label=pid, density=True)
plt.title("Noise distribution (low dose minus full dose), per patient")
plt.xlabel("HU difference")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
path = plot_dir / "04_noise_distribution_per_patient.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)

# --- 6e. noise standard deviation per patient ---
plt.figure(figsize=(7, 4))
plt.bar(noise_table["patient"], noise_table["noise_std_HU"], color="indianred")
plt.title("Noise level per patient (std of low minus full, HU)")
plt.xlabel("Patient")
plt.ylabel("Std (HU)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
path = plot_dir / "05_noise_std_per_patient.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)

print(noise_table)


# ############################################################################
# STEP 7: MODEL (RED-CNN) AND DATASET
# ############################################################################
class REDCNN(nn.Module):
    def __init__(self, in_channels=1, n_filters=64, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.enc1 = nn.Conv2d(in_channels, n_filters, kernel_size, padding=pad)
        self.enc2 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=pad)
        self.enc3 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=pad)
        self.enc4 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=pad)
        self.enc5 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=pad)
        self.dec5 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size, padding=pad)
        self.dec4 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size, padding=pad)
        self.dec3 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size, padding=pad)
        self.dec2 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size, padding=pad)
        self.dec1 = nn.ConvTranspose2d(n_filters, in_channels, kernel_size, padding=pad)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        e4 = self.relu(self.enc4(e3))
        e5 = self.relu(self.enc5(e4))
        d5 = self.relu(self.dec5(e5) + e4)
        d4 = self.relu(self.dec4(d5) + e3)
        d3 = self.relu(self.dec3(d4) + e2)
        d2 = self.relu(self.dec2(d3) + e1)
        d1 = self.dec1(d2)
        return self.relu(d1 + residual)


class CTDataset(Dataset):
    def __init__(self, recon_dir, patch_size=64, patches_per_slice=4):
        low_files = sorted((recon_dir / "low_dose").glob("*.npy"))
        full_names = {f.name for f in (recon_dir / "full_dose").glob("*.npy")}
        self.low_files = [f for f in low_files if f.name in full_names]    # only slices that exist in both
        self.full_files = [recon_dir / "full_dose" / f.name for f in self.low_files]
        self.patch_size = patch_size
        self.pps = patches_per_slice

    def __len__(self):
        return len(self.low_files) * self.pps

    def __getitem__(self, idx):
        slice_idx = idx // self.pps
        low = np.load(self.low_files[slice_idx]).astype(np.float32)
        full = np.load(self.full_files[slice_idx]).astype(np.float32)

        lo, hi = full.min(), full.max()                     # min-max scaling per slice
        low = (low - lo) / (hi - lo + 1e-8)
        full = (full - lo) / (hi - lo + 1e-8)

        h, w = full.shape
        ps = self.patch_size
        top = np.random.randint(0, max(1, h - ps))
        left = np.random.randint(0, max(1, w - ps))
        return (torch.from_numpy(low[top:top + ps, left:left + ps]).unsqueeze(0),
                torch.from_numpy(full[top:top + ps, left:left + ps]).unsqueeze(0))


# ############################################################################
# STEP 8: TRAIN (tracks MSE and PSNR per epoch, for train and validation)
# ############################################################################
dataset = CTDataset(recon_dir, patch_size=PATCH_SIZE, patches_per_slice=PATCHES_PER_SLICE)
n_val = max(1, int(0.1 * len(dataset)))
train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
print(f"Training patches: {len(train_set)}  |  Validation patches: {len(val_set)}")

model = REDCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

history = {"train_mse": [], "val_mse": [], "train_psnr": [], "val_psnr": []}
best_val_psnr = -float("inf")

for epoch in range(1, N_EPOCHS + 1):

    # ---- training pass ----
    model.train()
    mse_sum, psnr_sum, n_patches = 0.0, 0.0, 0
    for low, full in tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS} [train]"):
        low, full = low.to(device), full.to(device)
        optimizer.zero_grad()
        pred = model(low)
        loss = criterion(pred, full)
        loss.backward()
        optimizer.step()

        ref_np = full.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        for i in range(ref_np.shape[0]):
            data_range = ref_np[i, 0].max() - ref_np[i, 0].min()
            if data_range > 0:
                psnr_sum += peak_signal_noise_ratio(ref_np[i, 0], pred_np[i, 0], data_range=data_range)
                n_patches += 1
        mse_sum += loss.item() * low.size(0)

    history["train_mse"].append(mse_sum / len(train_set))
    history["train_psnr"].append(psnr_sum / max(1, n_patches))

    # ---- validation pass ----
    model.eval()
    mse_sum, psnr_sum, n_patches = 0.0, 0.0, 0
    with torch.no_grad():
        for low, full in val_loader:
            low, full = low.to(device), full.to(device)
            pred = model(low)
            mse_sum += criterion(pred, full).item() * low.size(0)

            ref_np = full.cpu().numpy()
            pred_np = pred.cpu().numpy()
            for i in range(ref_np.shape[0]):
                data_range = ref_np[i, 0].max() - ref_np[i, 0].min()
                if data_range > 0:
                    psnr_sum += peak_signal_noise_ratio(ref_np[i, 0], pred_np[i, 0], data_range=data_range)
                    n_patches += 1

    history["val_mse"].append(mse_sum / len(val_set))
    history["val_psnr"].append(psnr_sum / max(1, n_patches))

    log.info("Epoch %d | train MSE %.5f | val MSE %.5f | train PSNR %.2f dB | val PSNR %.2f dB",
             epoch, history["train_mse"][-1], history["val_mse"][-1],
             history["train_psnr"][-1], history["val_psnr"][-1])

    if history["val_psnr"][-1] > best_val_psnr:
        best_val_psnr = history["val_psnr"][-1]
        torch.save(model.state_dict(), model_dir / "best.pt")

history_table = pd.DataFrame(history, index=range(1, N_EPOCHS + 1))
history_table.index.name = "epoch"
history_table.to_csv(output_dir / "training_history.csv")
print(history_table.round(5))

# load the best checkpoint for evaluation
model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
model.eval()
print(f"Best validation PSNR: {best_val_psnr:.2f} dB")


# ############################################################################
# STEP 9: TRAINING CURVES (epochs vs MSE, epochs vs PSNR)
# ############################################################################
epochs = list(range(1, N_EPOCHS + 1))

# --- epochs vs MSE ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_mse"], marker="o", label="Training MSE", color="blue", linewidth=2)
plt.plot(epochs, history["val_mse"], marker="s", label="Validation MSE", color="orange", linewidth=2)
plt.title("RED-CNN: Epoch vs MSE (lower is better)", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
path = plot_dir / "06_epoch_vs_mse.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)

# --- epochs vs PSNR (the accuracy equivalent) ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_psnr"], marker="o", label="Training PSNR", color="blue", linewidth=2)
plt.plot(epochs, history["val_psnr"], marker="s", label="Validation PSNR", color="orange", linewidth=2)
plt.title("RED-CNN: Epoch vs PSNR (higher is better)", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
path = plot_dir / "07_epoch_vs_psnr.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)


# ############################################################################
# STEP 10: DENOISE SLICES, COMPARE IMAGES (target / input / output / error)
# ############################################################################
low_files = sorted((recon_dir / "low_dose").glob("*.npy"))
full_files = [recon_dir / "full_dose" / f.name for f in low_files]

# early, middle and late slice of the whole set
sample_indices = [len(low_files) // 4, len(low_files) // 2, len(low_files) - 1]
column_titles = ["Full dose (target)", "Low dose (input)", "AI denoised (output)", "Absolute error"]

fig, axes = plt.subplots(len(sample_indices), 4, figsize=(14, 3.6 * len(sample_indices)))
for row, idx in enumerate(sample_indices):
    low = np.load(low_files[idx]).astype(np.float32)
    full = np.load(full_files[idx]).astype(np.float32)
    lo, hi = full.min(), full.max()
    low_norm = (low - lo) / (hi - lo + 1e-8)
    full_norm = (full - lo) / (hi - lo + 1e-8)

    with torch.no_grad():
        x = torch.from_numpy(low_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        denoised = model(x).cpu().numpy()[0, 0]

    error_map = np.abs(full_norm - denoised)
    images = [full_norm, low_norm, denoised, error_map]
    cmaps = ["gray", "gray", "gray", "inferno"]

    for col in range(4):
        axes[row, col].imshow(images[col], cmap=cmaps[col])
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(column_titles[col], fontsize=11, fontweight="bold")
    axes[row, 0].text(5, 20, low_files[idx].stem, color="yellow", fontsize=8)

plt.tight_layout()
path = plot_dir / "08_multi_slice_comparison.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)

# --- single sample slice with its PSNR / SSIM printed (last slice) ---
idx = len(low_files) - 1
low = np.load(low_files[idx]).astype(np.float32)
full = np.load(full_files[idx]).astype(np.float32)
lo, hi = full.min(), full.max()
low_norm = (low - lo) / (hi - lo + 1e-8)
full_norm = (full - lo) / (hi - lo + 1e-8)
with torch.no_grad():
    x = torch.from_numpy(low_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    denoised = model(x).cpu().numpy()[0, 0]

rng = full_norm.max() - full_norm.min()
low_psnr = peak_signal_noise_ratio(full_norm, low_norm, data_range=rng)
low_ssim = structural_similarity(full_norm, low_norm, data_range=rng)
den_psnr = peak_signal_noise_ratio(full_norm, denoised, data_range=rng)
den_ssim = structural_similarity(full_norm, denoised, data_range=rng)
print(f"Sample slice {low_files[idx].stem}")
print(f"  Low dose : PSNR={low_psnr:.2f} dB  SSIM={low_ssim:.4f}")
print(f"  Denoised : PSNR={den_psnr:.2f} dB  SSIM={den_ssim:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
panels = [(f"Full dose (reference)", full_norm),
          (f"Low dose\nPSNR {low_psnr:.2f} dB, SSIM {low_ssim:.3f}", low_norm),
          (f"Low dose + AI denoise\nPSNR {den_psnr:.2f} dB, SSIM {den_ssim:.3f}", denoised)]
for ax, (title, img) in zip(axes, panels):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
path = plot_dir / "09_sample_slice_result.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)


# ############################################################################
# STEP 11: COMPARE METRICS (MSE, PSNR, SSIM) LOW DOSE vs DENOISED, PER PATIENT
# ############################################################################
rows = []
for f_low in tqdm(low_files[::EVAL_EVERY_N_SLICES], desc="Evaluating"):
    low = np.load(f_low).astype(np.float32)
    full = np.load(recon_dir / "full_dose" / f_low.name).astype(np.float32)
    lo, hi = full.min(), full.max()
    low_norm = (low - lo) / (hi - lo + 1e-8)
    full_norm = (full - lo) / (hi - lo + 1e-8)

    with torch.no_grad():
        x = torch.from_numpy(low_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        denoised = model(x).cpu().numpy()[0, 0]

    rng = full_norm.max() - full_norm.min()
    rows.append({
        "patient": f_low.stem.rsplit("_", 1)[0],
        "slice": f_low.stem,
        "low_mse": float(np.mean((full_norm - low_norm) ** 2)),
        "den_mse": float(np.mean((full_norm - denoised) ** 2)),
        "low_psnr": peak_signal_noise_ratio(full_norm, low_norm, data_range=rng),
        "den_psnr": peak_signal_noise_ratio(full_norm, denoised, data_range=rng),
        "low_ssim": structural_similarity(full_norm, low_norm, data_range=rng),
        "den_ssim": structural_similarity(full_norm, denoised, data_range=rng),
    })

metrics = pd.DataFrame(rows)
metrics.to_csv(output_dir / "slice_metrics.csv", index=False)
per_patient = metrics.drop(columns="slice").groupby("patient").mean()
per_patient.loc["ALL (mean)"] = metrics.drop(columns=["slice", "patient"]).mean()
per_patient.to_csv(output_dir / "metrics_per_patient.csv")
print(per_patient.round(4))

# --- grouped bar charts: low dose vs denoised, per patient ---
patient_only = per_patient.drop(index="ALL (mean)")
x_pos = np.arange(len(patient_only))
width = 0.38

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, low_col, den_col, better) in zip(
        axes, [("MSE", "low_mse", "den_mse", "lower is better"),
               ("PSNR (dB)", "low_psnr", "den_psnr", "higher is better"),
               ("SSIM", "low_ssim", "den_ssim", "higher is better")]):
    ax.bar(x_pos - width / 2, patient_only[low_col], width, label="Low dose", color="tab:red")
    ax.bar(x_pos + width / 2, patient_only[den_col], width, label="Denoised", color="tab:green")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(patient_only.index)
    ax.set_title(f"{name} ({better})")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
plt.tight_layout()
path = plot_dir / "10_metrics_per_patient.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)

# --- distribution of per-slice PSNR gain ---
plt.figure(figsize=(8, 4.5))
plt.hist(metrics["den_psnr"] - metrics["low_psnr"], bins=30, color="teal", alpha=0.8)
plt.axvline(0, color="black", linestyle="--")
plt.title("PSNR gain per slice (denoised minus low dose)")
plt.xlabel("PSNR gain (dB)")
plt.ylabel("Slices")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
path = plot_dir / "11_psnr_gain_distribution.png"
plt.savefig(path, dpi=150)
plt.show()
saved_plots.append(path)


# ############################################################################
# STEP 12: FINAL RESULTS
# ############################################################################
overall = per_patient.loc["ALL (mean)"]
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Patients used            : {len(patient_ids)}  {patient_ids}")
print(f"Total paired slices      : {int(slice_table['n_slices'].sum())}")
print(f"Epochs trained           : {N_EPOCHS}")
print(f"Best validation PSNR     : {best_val_psnr:.2f} dB")
print(f"Final validation MSE     : {history['val_mse'][-1]:.5f}")
print(f"Evaluated slices         : {len(metrics)}")
print("-" * 60)
print(f"{'Metric':<8}{'Low dose':>12}{'Denoised':>12}{'Change':>12}")
print(f"{'MSE':<8}{overall['low_mse']:>12.5f}{overall['den_mse']:>12.5f}{overall['den_mse'] - overall['low_mse']:>12.5f}")
print(f"{'PSNR':<8}{overall['low_psnr']:>12.2f}{overall['den_psnr']:>12.2f}{overall['den_psnr'] - overall['low_psnr']:>12.2f}")
print(f"{'SSIM':<8}{overall['low_ssim']:>12.4f}{overall['den_ssim']:>12.4f}{overall['den_ssim'] - overall['low_ssim']:>12.4f}")
print("=" * 60)


# ############################################################################
# STEP 13: SHOW ALL SAVED IMAGES AGAIN AND PRINT DOWNLOAD LINKS
# ############################################################################
for path in saved_plots:
    print(path.name)
    display(Image(filename=str(path)))

archive = shutil.make_archive("/kaggle/working/pipeline_outputs", "zip", output_dir)
print("Zip of everything:", archive)

# Kaggle FileLink paths are relative to /kaggle/working
print("\nDownload links:")
for path in saved_plots:
    display(FileLink(str(path.relative_to("/kaggle/working"))))
for name in ["training_history.csv", "slice_metrics.csv", "metrics_per_patient.csv", "model/best.pt"]:
    display(FileLink("outputs/" + name))
display(FileLink("pipeline_outputs.zip"))
