
# ============================================================================
# 0. CONFIGURATION
# ============================================================================
MANIFEST_PATH = "/kaggle/input/datasets/binishbatool/low-ct-dataa/LDCT-and-Projection-data_v07_20260520_Chest.tcia"
DOWNLOAD_DIR = "/kaggle/working/tcia_data"
OUTPUT_DIR = "/kaggle/working/outputs"
DRY_RUN = False          # set False for pass 2, after checking pass 1's report
PATIENT_LIMIT = 5       # caps how many patients' worth of data to download
N_EPOCHS = 5 #30
PATCH_SIZE = 45 #64
BATCH_SIZE = 16 #16

# ============================================================================
# 1. SETUP
# ============================================================================
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tcia_utils", "pydicom"], check=True)

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tcia_utils import nbia

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

print("Setup complete.")


# ============================================================================
# 2. METADATA QUERY — no download, just find out what's actually there
# ============================================================================

@dataclass
class SinogramPair:
    patient_id: str
    full_dose_uid: str
    low_dose_uid: str


FULL_DOSE_HINTS = ("full", "normal", "ndct", "high", "target", "1e5", "1e6", "hd")
LOW_DOSE_HINTS = ("low", "quarter", "ldct", "input", "1e4", "5e4", "ld", "25pct", "25%")
PROJECTION_HINTS = ("projection",)  # exclude — DICOM-CT-PD format, not standard-readable


def _classify_dose(text: str):
    t = text.lower()
    if any(h in t for h in LOW_DOSE_HINTS):
        return "low"
    if any(h in t for h in FULL_DOSE_HINTS):
        return "full"
    return None


def analyze_manifest(manifest_path: Path) -> tuple[pd.DataFrame, list[SinogramPair]]:
    """
    Parses the manifest into series UIDs, queries metadata for all of them
    (fast — no image download), filters out raw projection series, and
    pairs full-dose/low-dose image series by patient. Returns the filtered
    metadata dataframe and the resulting pairs — nothing is downloaded here.
    """
    log.info("Parsing manifest: %s", manifest_path)
    all_uids = nbia.manifestToList(str(manifest_path))
    log.info("Manifest lists %d total series", len(all_uids))

    log.info("Querying metadata for all series (no download) ...")
    meta = nbia.getSeriesList(uids=all_uids)
    if meta is None or len(meta) == 0:
        raise RuntimeError("Metadata query returned nothing — check TCIA connectivity.")

    desc_col = "SeriesDescription" if "SeriesDescription" in meta.columns else "Series Description"
    patient_col = "PatientID" if "PatientID" in meta.columns else "Patient ID"
    size_col = "FileSize" if "FileSize" in meta.columns else "File Size"

    is_projection = meta[desc_col].astype(str).str.contains(
        "|".join(PROJECTION_HINTS), case=False, na=False
    )
    image_meta = meta[~is_projection].copy()
    image_meta["_dose"] = image_meta[desc_col].astype(str).apply(_classify_dose)
    image_meta = image_meta[image_meta["_dose"].notna()]

    log.info("%d/%d series are non-projection with a recognizable dose label",
              len(image_meta), len(meta))

    tagged: dict[str, dict[str, list[str]]] = {}
    for _, row in image_meta.iterrows():
        patient_id = str(row[patient_col])
        dose = row["_dose"]
        uid = str(row["SeriesInstanceUID"])
        tagged.setdefault(patient_id, {"full": [], "low": []})[dose].append(uid)

    pairs = []
    for patient_id, doses in sorted(tagged.items()):
        n = min(len(doses["full"]), len(doses["low"]))
        for i in range(n):
            pairs.append(SinogramPair(patient_id, doses["full"][i], doses["low"][i]))

    return image_meta, pairs


def report(image_meta: pd.DataFrame, pairs: list[SinogramPair], patient_limit: int):
    size_col = "FileSize" if "FileSize" in image_meta.columns else "File Size"
    limited_patients = sorted({p.patient_id for p in pairs})[:patient_limit]
    limited_pairs = [p for p in pairs if p.patient_id in limited_patients]
    limited_uids = {p.full_dose_uid for p in limited_pairs} | {p.low_dose_uid for p in limited_pairs}
    limited_size_bytes = image_meta[image_meta["SeriesInstanceUID"].isin(limited_uids)][size_col].sum()

    all_uids = {p.full_dose_uid for p in pairs} | {p.low_dose_uid for p in pairs}
    all_size_bytes = image_meta[image_meta["SeriesInstanceUID"].isin(all_uids)][size_col].sum()

    print(f"\n{'='*60}")
    print(f"Total patients with a usable full+low image pair: {len(set(p.patient_id for p in pairs))}")
    print(f"Total pairs: {len(pairs)}  |  Total size if downloading ALL: {all_size_bytes / 1e9:.2f} GB")
    print(f"\nWith PATIENT_LIMIT={patient_limit}:")
    print(f"  Patients: {len(limited_patients)}  |  Pairs: {len(limited_pairs)}")
    print(f"  Series to download: {len(limited_uids)}  |  Size: {limited_size_bytes / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    if limited_size_bytes / 1e9 > 15:
        print("WARNING: this is close to or over Kaggle's ~20GB working-directory "
              "quota. Lower PATIENT_LIMIT before setting DRY_RUN=False.")
    return limited_patients, limited_pairs


# ============================================================================
# 3. DOWNLOAD — only the filtered, size-checked series
# ============================================================================

def download_pairs(pairs: list[SinogramPair], out_dir: Path) -> None:
    uids = list({p.full_dose_uid for p in pairs} | {p.low_dose_uid for p in pairs})
    log.info("Downloading %d filtered series (not the full manifest) ...", len(uids))
    nbia.downloadSeries(uids, input_type="list", path=str(out_dir))
    log.info("Download complete.")


def _find_dicom_files(series_dir: Path):
    candidates = []
    for pattern in ("*.dcm", "*.DCM", "*.IMA", "*.ima"):
        candidates.extend(series_dir.glob(pattern))
    if not candidates:
        for f in series_dir.iterdir():
            if f.is_file() and f.suffix == "":
                try:
                    with open(f, "rb") as fh:
                        fh.seek(128)
                        if fh.read(4) == b"DICM":
                            candidates.append(f)
                except OSError:
                    continue
    return sorted(candidates)


def load_dicom_series(series_dir: Path) -> np.ndarray:
    dicom_files = _find_dicom_files(series_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM-like files found in {series_dir}")
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    return volume * slope + intercept


def export_pairs_to_npy(pairs: list[SinogramPair], download_dir: Path, out_dir: Path) -> None:
    """NBIA downloads each series into download_dir/<SeriesInstanceUID>/."""
    low_out = out_dir / "low_dose"
    full_out = out_dir / "full_dose"
    low_out.mkdir(parents=True, exist_ok=True)
    full_out.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        low_dir = download_dir / pair.low_dose_uid
        full_dir = download_dir / pair.full_dose_uid
        low_vol = load_dicom_series(low_dir)
        full_vol = load_dicom_series(full_dir)
        n_slices = min(low_vol.shape[0], full_vol.shape[0])
        for i in range(n_slices):
            np.save(low_out / f"{pair.patient_id}_{i:04d}.npy", low_vol[i])
            np.save(full_out / f"{pair.patient_id}_{i:04d}.npy", full_vol[i])
        log.info("Exported %d slices for patient %s", n_slices, pair.patient_id)


# ============================================================================
# 4. MODEL, METRICS, TRAINING (unchanged from before — only the data-getting
#    steps above were the bug)
# ============================================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt


class REDCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_filters: int = 96, kernel_size: int = 5):
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
        residual_input = x
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
        return self.relu(d1 + residual_input)


def compute_psnr(reference, test):
    data_range = reference.max() - reference.min()
    return float(peak_signal_noise_ratio(reference, test, data_range=data_range))


def compute_ssim(reference, test):
    data_range = reference.max() - reference.min()
    return float(structural_similarity(reference, test, data_range=data_range))


class LowDosePatchDataset(Dataset):
    def __init__(self, recon_dir: Path, patch_size: int = 64, patches_per_slice: int = 8):
        self.low_dose_files = sorted((recon_dir / "low_dose").glob("*.npy"))
        self.full_dose_files = sorted((recon_dir / "full_dose").glob("*.npy"))
        assert len(self.low_dose_files) == len(self.full_dose_files)
        self.patch_size = patch_size
        self.patches_per_slice = patches_per_slice

    def __len__(self):
        return len(self.low_dose_files) * self.patches_per_slice

    def __getitem__(self, idx):
        slice_idx = idx // self.patches_per_slice
        low = np.load(self.low_dose_files[slice_idx]).astype(np.float32)
        full = np.load(self.full_dose_files[slice_idx]).astype(np.float32)
        lo, hi = full.min(), full.max()
        low = (low - lo) / (hi - lo + 1e-8)
        full = (full - lo) / (hi - lo + 1e-8)
        h, w = full.shape
        ps = self.patch_size
        top = np.random.randint(0, max(1, h - ps))
        left = np.random.randint(0, max(1, w - ps))
        return (torch.from_numpy(low[top:top+ps, left:left+ps]).unsqueeze(0),
                torch.from_numpy(full[top:top+ps, left:left+ps]).unsqueeze(0))


def train_model(recon_dir: Path, out_dir: Path, epochs: int, batch_size: int, patch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Training on device: %s", device)

    dataset = LowDosePatchDataset(recon_dir, patch_size=patch_size)
    n_val = max(1, int(0.1 * len(dataset)))
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = REDCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    best_val_psnr = -float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for low, full in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            low, full = low.to(device), full.to(device)
            optimizer.zero_grad()
            pred = model(low)
            loss = criterion(pred, full)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * low.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_psnr, val_ssim, n = 0.0, 0.0, 0
        with torch.no_grad():
            for low, full in val_loader:
                low, full = low.to(device), full.to(device)
                pred = model(low).cpu().numpy()
                full_np = full.cpu().numpy()
                for i in range(pred.shape[0]):
                    val_psnr += compute_psnr(full_np[i, 0], pred[i, 0])
                    val_ssim += compute_ssim(full_np[i, 0], pred[i, 0])
                    n += 1
        val_psnr /= n
        val_ssim /= n
        log.info("Epoch %d: train_loss=%.5f val_psnr=%.2fdB val_ssim=%.4f",
                  epoch, train_loss, val_psnr, val_ssim)

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), out_dir / "best.pt")

    log.info("Training complete. Best val PSNR: %.2fdB", best_val_psnr)
    return model


# ============================================================================
# 5. RUN
# ============================================================================

manifest_path = Path(MANIFEST_PATH)
download_dir = Path(DOWNLOAD_DIR)
output_dir = Path(OUTPUT_DIR)
recon_dir = output_dir / "recon"
model_dir = output_dir / "model"

if not manifest_path.exists():
    raise FileNotFoundError(f"Manifest not found at {manifest_path}")

image_meta, all_pairs = analyze_manifest(manifest_path)
limited_patients, pairs = report(image_meta, all_pairs, PATIENT_LIMIT)

if DRY_RUN:
    print("DRY_RUN=True — stopping here. Review the report above, adjust "
          "PATIENT_LIMIT if needed, then set DRY_RUN=False and re-run.")
else:
    print(f"Downloading {len(pairs)} pairs for {len(limited_patients)} patients ...")
    download_pairs(pairs, download_dir)

    print("Exporting paired slices ...")
    export_pairs_to_npy(pairs, download_dir, recon_dir)

    print("Training ...")
    model = train_model(recon_dir, model_dir, epochs=N_EPOCHS, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE)

    print("Evaluating on a sample slice ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    low_files = sorted((recon_dir / "low_dose").glob("*.npy"))
    full_files = sorted((recon_dir / "full_dose").glob("*.npy"))
    sample_idx = len(low_files) - 1

    low = np.load(low_files[sample_idx]).astype(np.float32)
    full = np.load(full_files[sample_idx]).astype(np.float32)
    lo, hi = full.min(), full.max()
    low_norm = (low - lo) / (hi - lo + 1e-8)
    full_norm = (full - lo) / (hi - lo + 1e-8)

    model.eval()
    with torch.no_grad():
        low_t = torch.from_numpy(low_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        denoised = model(low_t).cpu().numpy()[0, 0]

    print(f"Low-dose:  PSNR={compute_psnr(full_norm, low_norm):.2f}dB  SSIM={compute_ssim(full_norm, low_norm):.4f}")
    print(f"Denoised:  PSNR={compute_psnr(full_norm, denoised):.2f}dB  SSIM={compute_ssim(full_norm, denoised):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, title, img in zip(axes, ["Full-dose\n(reference)", "Low-dose\n(input)", "Low-dose + AI denoise"],
                               [full_norm, low_norm, denoised]):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    fig_path = output_dir / "tcia_pipeline_result.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved comparison figure to {fig_path}")
    print("\nDone. Model checkpoint:", model_dir / "best.pt")
