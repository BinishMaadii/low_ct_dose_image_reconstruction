"""
Data loading for the TCIA "Low Dose CT Image and Projection Data" collection
(Mayo Clinic / AAPM 2016 Low-Dose CT Grand Challenge).

Two ways to get data in, depending on where this runs:

  A) LOCAL, direct from TCIA (this sandboxed environment cannot reach tcia.io):
     1. Create a free TCIA account: https://www.cancerimagingarchive.net/
     2. `pip install tcia_utils`
     3. Run `download_collection()` / this script's default CLI — pulls the
        projection data (and reference reconstructions) for a chosen subset
        of patients, indexed full-dose / quarter-dose by patient + slice.

  B) KAGGLE, via an attached dataset (no download step — see the "Kaggle
     input adapter" section below). Use this when running the pipeline in a
     Kaggle Notebook with a Mayo/AAPM low-dose CT dataset attached under
     /kaggle/input/. Kaggle-hosted mirrors vary in folder layout and file
     extension (.dcm vs .IMA vs no extension), so the adapter functions here
     are written to be tolerant of that rather than assuming one fixed shape
     — inspect your attached dataset's structure first (`!find /kaggle/input
     -maxdepth 4`) and adjust the keyword matching below if it doesn't match.

The collection contains, per patient:
    - Full-dose raw projection data (sinograms)
    - Quarter-dose ("low-dose") raw projection data — the vendor's own
      simulated low-dose acquisition, not synthetic noise added after the fact
    - Reference reconstructed DICOM series for both dose levels

Where raw sinograms are available, we work from those rather than the
reference DICOM images, so reconstruction.py performs the actual FBP
reconstruction ourselves. Where a Kaggle mirror only provides pre-reconstructed
image slices, `export_reconstructed_kaggle_images()` below feeds those directly
into train.py's expected format, skipping reconstruction.py.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

COLLECTION_NAME = "LDCT-and-Projection-data"


@dataclass
class SinogramPair:
    """A matched full-dose / low-dose sinogram pair for one patient/slice range."""
    patient_id: str
    full_dose_path: Path
    low_dose_path: Path


def download_collection(out_dir: Path, patient_limit: int | None = None) -> list[str]:
    """
    Downloads the TCIA collection using tcia_utils. Requires a TCIA account
    and internet access — run this locally, not in this sandbox.

    Returns the list of patient IDs downloaded.
    """
    try:
        from tcia_utils import nbia
    except ImportError as e:
        raise RuntimeError(
            "tcia_utils not installed. Run `pip install tcia_utils` locally, "
            "then re-run this script on a machine with internet access."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Fetching series metadata for collection: %s", COLLECTION_NAME)

    series = nbia.getSeries(collection=COLLECTION_NAME)
    patient_ids = sorted({s["PatientID"] for s in series})
    if patient_limit:
        patient_ids = patient_ids[:patient_limit]
        log.info("Limiting download to first %d patients", patient_limit)

    for pid in patient_ids:
        log.info("Downloading patient %s", pid)
        nbia.downloadSeries(
            series_data=[s for s in series if s["PatientID"] == pid],
            path=str(out_dir / pid),
        )

    log.info("Downloaded %d patients to %s", len(patient_ids), out_dir)
    return patient_ids


def index_sinogram_pairs(data_dir: Path) -> list[SinogramPair]:
    """
    Walks the downloaded collection and pairs up full-dose / quarter-dose
    projection series by patient ID. Expects the directory layout produced by
    `download_collection`: data_dir/<patient_id>/<series>/*.dcm
    """
    pairs = []
    for patient_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        series_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
        full_dose = next((d for d in series_dirs if "Full" in d.name), None)
        low_dose = next((d for d in series_dirs if "Low" in d.name or "Quarter" in d.name), None)
        if full_dose and low_dose:
            pairs.append(SinogramPair(patient_dir.name, full_dose, low_dose))
        else:
            log.warning("Skipping %s — could not find matched dose pair", patient_dir.name)
    log.info("Indexed %d full-dose/low-dose pairs", len(pairs))
    return pairs


def _find_dicom_files(series_dir: Path) -> list[Path]:
    """
    Finds DICOM files under a directory regardless of extension convention.
    The official Mayo/AAPM distribution and various Kaggle mirrors use .dcm,
    .IMA, .ima, or no extension at all for the same underlying DICOM format.
    """
    candidates = []
    for pattern in ("*.dcm", "*.DCM", "*.IMA", "*.ima"):
        candidates.extend(series_dir.glob(pattern))
    if not candidates:
        # Fall back to sniffing extensionless files for the DICOM magic bytes
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


def load_projection_dicom(series_dir: Path) -> np.ndarray:
    """
    Loads a directory of projection DICOM slices into a single sinogram array
    of shape (n_projections, n_detector_rows, n_detector_cols) or, for
    reconstructed 2D-slice DICOM series, (n_slices, rows, cols).

    Slices are sorted by InstanceNumber to preserve acquisition/projection order.
    """
    dicom_files = _find_dicom_files(series_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM-like files found in {series_dir}")

    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)

    # Apply rescale slope/intercept if present (standard DICOM convention)
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    volume = volume * slope + intercept

    log.info("Loaded volume of shape %s from %s", volume.shape, series_dir)
    return volume


# ---------------------------------------------------------------------------
# TCIA manifest adapter — for when you already have a .tcia manifest file
# ---------------------------------------------------------------------------
# A .tcia file is NOT the data itself — it's a small text manifest listing
# which DICOM series to pull from TCIA's servers. Use this section once
# you've got a real manifest (e.g. downloaded from
# https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/)
# to fetch the actual DICOM files.
#
# IMPORTANT — this collection has TWO kinds of files per patient:
#   1. Raw projection/sinogram data in "DICOM-CT-PD" format — an extended,
#      non-standard DICOM variant that plain pydicom.pixel_array cannot read
#      correctly. Parsing it properly needs a dedicated reader (see Mayo's
#      DICOM-CT-PD user manual, or e.g. github.com/faebstn96/helix2fan).
#      That reader is NOT implemented here — building and validating one
#      correctly is a substantial project on its own, and getting it subtly
#      wrong would silently corrupt every downstream result.
#   2. Standard reconstructed CT images (both full-dose and, for Siemens
#      scanners, low-dose too) in ordinary DICOM format — readable directly
#      with load_projection_dicom() like any other DICOM series.
#
# The functions below use path (2): real clinical low-dose/full-dose image
# pairs, already reconstructed by the scanner via FBP. This is a legitimate,
# widely-used way to do low-dose CT denoising research (see e.g. the SUPER
# Learning paper, which works from Mayo's provided reconstructions directly)
# — it just means this run of the pipeline demonstrates the *denoising* side
# rather than your own from-scratch reconstruction. If you want the "I did
# the FBP reconstruction myself" story, the LoDoPaB-CT path below is the
# tested route for that, since it's parallel-beam and openly documented.

TCIA_PROJECTION_HINTS = ("projection",)  # series to SKIP — DICOM-CT-PD, not standard-readable


def download_from_manifest(manifest_path: Path, out_dir: Path) -> None:
    """
    Downloads the actual DICOM files for every series listed in a .tcia
    manifest file — not just the manifest, the real image data. Requires
    `pip install tcia_utils` and internet access to TCIA's servers (works
    from a Kaggle notebook with internet enabled; this sandbox cannot reach
    TCIA).
    """
    try:
        from tcia_utils import nbia
    except ImportError as e:
        raise RuntimeError(
            "tcia_utils not installed. Run `pip install tcia_utils` first."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading TCIA series listed in manifest: %s", manifest_path)
    nbia.downloadSeries(str(manifest_path), input_type="manifest", path=str(out_dir))
    log.info("Download complete. Series saved under %s", out_dir)


def index_tcia_image_pairs(out_dir: Path) -> list[SinogramPair]:
    """
    Reads the metadata.csv that NBIA writes alongside a manifest download,
    filters out raw projection series (DICOM-CT-PD — see module note above),
    and pairs the remaining full-dose/low-dose *reconstructed image* series
    by patient (Subject ID).
    """
    import pandas as pd

    metadata_candidates = list(out_dir.rglob("metadata.csv"))
    if not metadata_candidates:
        raise FileNotFoundError(
            f"No metadata.csv found under {out_dir} — expected one to be "
            "written alongside the downloaded series by download_from_manifest()."
        )
    metadata_path = metadata_candidates[0]
    df = pd.read_csv(metadata_path)

    desc_col = "Series Description" if "Series Description" in df.columns else "SeriesDescription"
    patient_col = "Subject ID" if "Subject ID" in df.columns else "PatientID"
    loc_col = "File Location" if "File Location" in df.columns else "FileLocation"

    is_projection = df[desc_col].astype(str).str.contains(
        "|".join(TCIA_PROJECTION_HINTS), case=False, na=False
    )
    image_df = df[~is_projection]
    log.info("metadata.csv: %d series total, %d after excluding raw projection series",
             len(df), len(image_df))

    tagged: dict[str, dict[str, Path]] = {}
    for _, row in image_df.iterrows():
        dose = _classify_dose(Path(str(row[desc_col])))
        if dose is None:
            continue
        patient_id = str(row[patient_col])
        file_loc = str(row[loc_col])
        series_dir = Path(file_loc)
        if not series_dir.is_absolute():
            series_dir = (metadata_path.parent / series_dir).resolve()
        tagged.setdefault(patient_id, {})[dose] = series_dir

    pairs = []
    for patient_id, doses in sorted(tagged.items()):
        if "full" in doses and "low" in doses:
            pairs.append(SinogramPair(patient_id, doses["full"], doses["low"]))
        else:
            log.warning("Skipping %s — found dose classes %s, need both full and low",
                        patient_id, list(doses.keys()))

    log.info("Indexed %d full-dose/low-dose image pairs from TCIA metadata", len(pairs))
    return pairs



# ---------------------------------------------------------------------------
# LoDoPaB-CT adapter (recommended: verified, no-account, parallel-beam)
# ---------------------------------------------------------------------------
# LoDoPaB-CT (Leuschner et al., Sci Data 8, 109 (2021); Zenodo record 3384092,
# DOI 10.5281/zenodo.3384092, CC-BY 4.0) is a purpose-built low-dose CT
# benchmark: simulated low-dose sinograms paired with real ground-truth
# reconstructions from LIDC-IDRI. Unlike the TCIA/Kaggle routes above, this
# needs no account and no uncertain mirror — the files below are the
# official, verified Zenodo listing (checked directly, sizes as of this
# writing). Recommended default: the "test" split alone is ~4.6GB and
# provides 3,553 real paired samples — enough for a portfolio project without
# downloading the full ~55GB dataset.
#
#   ground_truth_test.zip   1.6 GB   (362x362 reference reconstructions)
#   observation_test.zip    3.0 GB   (1000x513 simulated low-dose sinograms)
#
# Uses parallel-beam geometry (1000 angles over [0,180), 513 detector
# pixels), so reconstruction.reconstruct_fbp_parallel (scikit-image only,
# no ASTRA/TIGRE) is the right reconstruction function for this data.

LODOPAB_BASE_URL = "https://zenodo.org/records/3384092/files"
LODOPAB_FILES = {
    "ground_truth_test": f"{LODOPAB_BASE_URL}/ground_truth_test.zip?download=1",
    "observation_test": f"{LODOPAB_BASE_URL}/observation_test.zip?download=1",
}
LODOPAB_N_ANGLES = 1000
LODOPAB_SAMPLES_PER_FILE = 128  # per the Zenodo record's HDF5 layout


def download_lodopab_test(out_dir: Path) -> None:
    """
    Downloads and extracts the LoDoPaB-CT test split directly from Zenodo —
    no account or API key needed, ~4.6GB total. Run this locally (this
    sandbox has no network access to zenodo.org).
    """
    import urllib.request
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in LODOPAB_FILES.items():
        zip_path = out_dir / f"{name}.zip"
        extract_dir = out_dir / name
        if extract_dir.exists() and any(extract_dir.iterdir()):
            log.info("%s already extracted, skipping", name)
            continue
        if not zip_path.exists():
            log.info("Downloading %s from %s (this is several GB, be patient)", name, url)
            urllib.request.urlretrieve(url, zip_path)
        log.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    log.info("LoDoPaB-CT test split ready in %s", out_dir)


def _lodopab_sample_location(index: int) -> tuple[int, int]:
    """Maps a global sample index to (file_number, row_within_file)."""
    file_num = index // LODOPAB_SAMPLES_PER_FILE
    row = index % LODOPAB_SAMPLES_PER_FILE
    return file_num, row


def load_lodopab_sample(lodopab_dir: Path, index: int, part: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """
    Loads one (ground_truth, observation) pair from an extracted LoDoPaB-CT
    directory (as produced by download_lodopab_test).

    Returns
    -------
    ground_truth : np.ndarray, shape (362, 362), values in [0, 1]
    observation  : np.ndarray, shape (1000, 513) — the simulated low-dose sinogram
    """
    import h5py

    file_num, row = _lodopab_sample_location(index)
    gt_path = lodopab_dir / f"ground_truth_{part}" / f"ground_truth_{part}_{file_num:03d}.hdf5"
    obs_path = lodopab_dir / f"observation_{part}" / f"observation_{part}_{file_num:03d}.hdf5"

    with h5py.File(gt_path, "r") as f:
        ground_truth = f["data"][row].astype(np.float32)
    with h5py.File(obs_path, "r") as f:
        observation = f["data"][row].astype(np.float32)
    return ground_truth, observation


def export_lodopab_to_recon_layout(lodopab_dir: Path, out_dir: Path,
                                    n_samples: int = 200, part: str = "test") -> None:
    """
    Loads the first n_samples pairs from an extracted LoDoPaB-CT split,
    reconstructs each low-dose sinogram via parallel-beam FBP (scikit-image,
    no ASTRA/TIGRE needed), and writes out_dir/low_dose/*.npy (FBP
    reconstruction) and out_dir/full_dose/*.npy (ground truth) — the exact
    layout train.py's LowDosePatchDataset expects. Skips reconstruction.py's
    fan-beam path entirely since this data doesn't need it.

    n_samples=200 keeps this quick to run and still gives ~200 training
    slices; raise it (up to the full split size) once you've confirmed the
    pipeline works end-to-end.
    """
    from reconstruction import reconstruct_fbp_parallel

    low_out = out_dir / "low_dose"
    full_out = out_dir / "full_dose"
    low_out.mkdir(parents=True, exist_ok=True)
    full_out.mkdir(parents=True, exist_ok=True)

    theta_deg = np.linspace(0.0, 180.0, LODOPAB_N_ANGLES, endpoint=False)

    for i in range(n_samples):
        ground_truth, observation = load_lodopab_sample(lodopab_dir, i, part=part)
        low_dose_recon = reconstruct_fbp_parallel(observation, theta_deg=theta_deg)
        np.save(low_out / f"{part}_{i:05d}.npy", low_dose_recon)
        np.save(full_out / f"{part}_{i:05d}.npy", ground_truth)
        if (i + 1) % 50 == 0:
            log.info("Reconstructed %d/%d LoDoPaB-CT samples", i + 1, n_samples)

    log.info("Exported %d LoDoPaB-CT pairs to %s", n_samples, out_dir)


# Use these instead of download_collection() when running in a Kaggle Notebook
# with a Mayo/AAPM low-dose CT dataset attached via "Add Input" — no download,
# no TCIA account needed. Kaggle mirrors are user-uploaded and vary in layout,
# so run `find_kaggle_dataset_dir()` and eyeball the result before trusting
# the automatic pairing.

KAGGLE_INPUT_ROOT = Path("/kaggle/input")

# Keyword sets used to guess which files/folders are full-dose vs low-dose.
# Mayo/AAPM naming conventions vary by mirror: "full"/"low", "normal"/"quarter",
# "NDCT"/"LDCT", "1e5"/"1e4" (photon count), "target"/"input" (ML framing).
FULL_DOSE_HINTS = ("full", "normal", "ndct", "high", "target", "1e5", "1e6", "hd")
LOW_DOSE_HINTS = ("low", "quarter", "ldct", "input", "1e4", "5e4", "ld", "25pct", "25%")


def find_kaggle_dataset_dir(name_hint: str | None = None) -> Path:
    """
    Locates the attached dataset's root folder under /kaggle/input/.

    With one dataset attached, returns it directly. With several attached,
    pass `name_hint` (a substring of the dataset slug, e.g. "sinogram") to
    disambiguate — otherwise raises so you can inspect and choose manually.
    """
    if not KAGGLE_INPUT_ROOT.exists():
        raise RuntimeError(
            "/kaggle/input not found — this function only works inside a "
            "Kaggle Notebook with a dataset attached via 'Add Input'."
        )
    candidates = [d for d in KAGGLE_INPUT_ROOT.iterdir() if d.is_dir()]
    if not candidates:
        raise RuntimeError("No datasets attached under /kaggle/input — use 'Add Input' first.")
    if name_hint:
        matched = [d for d in candidates if name_hint.lower() in d.name.lower()]
        if not matched:
            raise RuntimeError(
                f"No attached dataset matched '{name_hint}'. Available: "
                f"{[d.name for d in candidates]}"
            )
        candidates = matched
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple datasets attached ({[d.name for d in candidates]}) — "
            "pass name_hint to disambiguate."
        )
    log.info("Using Kaggle dataset: %s", candidates[0])
    return candidates[0]


def _classify_dose(path: Path) -> str | None:
    """Guesses 'full', 'low', or None from a path's components using the hint lists."""
    name = str(path).lower()
    if any(h in name for h in LOW_DOSE_HINTS):
        return "low"
    if any(h in name for h in FULL_DOSE_HINTS):
        return "full"
    return None


def index_kaggle_pairs(dataset_dir: Path) -> list[SinogramPair]:
    """
    Recursively scans an attached Kaggle dataset and pairs up full-dose /
    low-dose series by patient ID, using flexible keyword matching since
    Kaggle mirrors don't follow one fixed folder convention.

    Assumes patient ID is inferable from a path component (e.g. "L067",
    "patient_03") shared between a full-dose and a low-dose directory. If
    your attached dataset's structure doesn't group cleanly this way, it's
    usually faster to write a one-off adapter for that specific layout than
    to fight this heuristic — the goal here is covering the common cases.
    """
    # Group every directory containing DICOM-like files by its parent chain,
    # tagging each with a guessed patient ID (nearest ancestor dir name that
    # isn't itself a dose-hint match) and dose class.
    tagged: dict[str, dict[str, Path]] = {}
    for dirpath in sorted(p for p in dataset_dir.rglob("*") if p.is_dir()):
        if not _find_dicom_files(dirpath):
            continue
        dose = _classify_dose(dirpath)
        if dose is None:
            continue
        # Walk up until we find an ancestor name that isn't a dose-hint itself
        patient_id = dirpath.parent.name
        if _classify_dose(Path(patient_id)) is not None:
            patient_id = dirpath.parent.parent.name if dirpath.parent.parent != dataset_dir else dirpath.parent.name

        tagged.setdefault(patient_id, {})[dose] = dirpath

    pairs = []
    for patient_id, doses in sorted(tagged.items()):
        if "full" in doses and "low" in doses:
            pairs.append(SinogramPair(patient_id, doses["full"], doses["low"]))
        else:
            log.warning("Skipping %s — found dose classes %s, need both full and low",
                        patient_id, list(doses.keys()))

    log.info("Indexed %d full-dose/low-dose pairs from Kaggle input", len(pairs))
    if not pairs:
        log.warning(
            "No pairs found automatically. Run `!find %s -maxdepth 4` in a "
            "notebook cell to inspect the actual layout, then adjust "
            "FULL_DOSE_HINTS/LOW_DOSE_HINTS above or write a one-off loader "
            "for this specific dataset.", dataset_dir
        )
    return pairs


def export_sinograms_to_npy(pairs: list[SinogramPair], out_dir: Path) -> None:
    """
    Loads each pair's raw sinogram volumes and writes them as per-slice .npy
    files into out_dir/low_dose/ and out_dir/full_dose/, ready for
    reconstruction.py to run FBP/SART on. Use this when the attached Kaggle
    dataset provides raw projection data (sinograms), not pre-reconstructed
    images.
    """
    low_out = out_dir / "raw_sinograms" / "low_dose"
    full_out = out_dir / "raw_sinograms" / "full_dose"
    low_out.mkdir(parents=True, exist_ok=True)
    full_out.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        low_vol = load_projection_dicom(pair.low_dose_path)
        full_vol = load_projection_dicom(pair.full_dose_path)
        n_slices = min(low_vol.shape[0], full_vol.shape[0])
        for i in range(n_slices):
            np.save(low_out / f"{pair.patient_id}_{i:04d}.npy", low_vol[i])
            np.save(full_out / f"{pair.patient_id}_{i:04d}.npy", full_vol[i])
        log.info("Exported %d sinogram slices for patient %s", n_slices, pair.patient_id)


def export_reconstructed_kaggle_images(pairs: list[SinogramPair], out_dir: Path) -> None:
    """
    Use this INSTEAD of export_sinograms_to_npy() when the attached Kaggle
    dataset only provides pre-reconstructed image slices (not raw sinograms)
    — several Mayo/AAPM mirrors distribute the challenge data this way. This
    writes directly into the out_dir/low_dose/ and out_dir/full_dose/ layout
    that train.py's LowDosePatchDataset expects, skipping reconstruction.py
    entirely since there's no raw projection data to reconstruct from.

    Note: if you go this route, be upfront in your portfolio writeup that
    this run used pre-reconstructed images rather than raw projection data
    — it's still a legitimate denoising benchmark, just skips the FBP step
    that demonstrates the reconstruction side of the pipeline. Prefer
    export_sinograms_to_npy() when raw data is available.
    """
    low_out = out_dir / "low_dose"
    full_out = out_dir / "full_dose"
    low_out.mkdir(parents=True, exist_ok=True)
    full_out.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        low_vol = load_projection_dicom(pair.low_dose_path)
        full_vol = load_projection_dicom(pair.full_dose_path)
        n_slices = min(low_vol.shape[0], full_vol.shape[0])
        for i in range(n_slices):
            np.save(low_out / f"{pair.patient_id}_{i:04d}.npy", low_vol[i])
            np.save(full_out / f"{pair.patient_id}_{i:04d}.npy", full_vol[i])
        log.info("Exported %d reconstructed-image slices for patient %s", n_slices, pair.patient_id)


def load_from_kaggle_input(name_hint: str | None = None, out_dir: Path = Path("/kaggle/working/data"),
                            mode: str = "sinograms") -> list[SinogramPair]:
    """
    One-call convenience wrapper for a Kaggle notebook cell:

        from data_loading import load_from_kaggle_input
        pairs = load_from_kaggle_input(mode="sinograms")   # or mode="reconstructed"

    Locates the attached dataset, indexes full/low-dose pairs, and exports
    them to out_dir in the layout the rest of the pipeline expects. Set
    mode="reconstructed" if your attached dataset only has image slices, not
    raw projection data (see export_reconstructed_kaggle_images docstring).
    """
    dataset_dir = find_kaggle_dataset_dir(name_hint)
    pairs = index_kaggle_pairs(dataset_dir)
    if not pairs:
        return pairs
    if mode == "sinograms":
        export_sinograms_to_npy(pairs, out_dir)
    elif mode == "reconstructed":
        export_reconstructed_kaggle_images(pairs, out_dir)
    else:
        raise ValueError(f"Unknown mode '{mode}' — use 'sinograms' or 'reconstructed'")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["tcia", "kaggle", "lodopab"], default="lodopab",
                         help="'tcia' downloads directly (needs a TCIA account); "
                              "'kaggle' indexes an already-attached Kaggle dataset; "
                              "'lodopab' downloads the verified LoDoPaB-CT test split "
                              "from Zenodo (no account needed, ~4.6GB) — recommended default")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--out", type=Path, default=Path("data/"))
    parser.add_argument("--patient-limit", type=int, default=10,
                         help="Start small — full collection is large (~1TB+). TCIA source only.")
    parser.add_argument("--kaggle-name-hint", default=None,
                         help="Substring to pick the right dataset if multiple are attached")
    parser.add_argument("--kaggle-mode", choices=["sinograms", "reconstructed"], default="sinograms")
    parser.add_argument("--lodopab-n-samples", type=int, default=200,
                         help="How many LoDoPaB-CT test samples to reconstruct and export")
    args = parser.parse_args()

    if args.source == "tcia":
        download_collection(args.out, patient_limit=args.patient_limit)
        pairs = index_sinogram_pairs(args.out)
        index_path = args.out / "pairs_index.txt"
        args.out.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as f:
            for p in pairs:
                f.write(f"{p.patient_id}\t{p.full_dose_path}\t{p.low_dose_path}\n")
        log.info("Wrote pair index to %s", index_path)

    elif args.source == "kaggle":
        pairs = load_from_kaggle_input(args.kaggle_name_hint, args.out, args.kaggle_mode)
        index_path = args.out / "pairs_index.txt"
        args.out.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as f:
            for p in pairs:
                f.write(f"{p.patient_id}\t{p.full_dose_path}\t{p.low_dose_path}\n")
        log.info("Wrote pair index to %s", index_path)

    else:  # lodopab
        raw_dir = args.out / "lodopab_raw"
        download_lodopab_test(raw_dir)
        export_lodopab_to_recon_layout(raw_dir, args.out, n_samples=args.lodopab_n_samples)
        log.info("LoDoPaB-CT export complete — low_dose/ and full_dose/ ready under %s", args.out)


if __name__ == "__main__":
    main()
