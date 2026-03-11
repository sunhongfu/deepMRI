"""
DICOM series → NIfTI conversion utilities for the iQSM+ web interface.

Handles single- and multi-echo GRE DICOM series (phase and magnitude).
Slice ordering, affine construction, and pixel rescaling are all handled
here so the rest of the pipeline sees ordinary NIfTI files.

Coordinate convention
---------------------
DICOM stores coordinates in LPS (Left-Posterior-Superior).
NIfTI / the iQSM+ pipeline expects RAS (Right-Anterior-Superior).
The affine built here applies the LPS→RAS flip (negate x and y columns)
so nibabel / the downstream code sees consistent orientations.
"""

import os
import tempfile

import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# pydicom import (optional dependency – only needed for DICOM input)
# ---------------------------------------------------------------------------

def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM input.  "
            "Install it with:  pip install pydicom"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(ds, tag, default=None):
    """Return getattr(ds, tag) or *default* if the attribute is absent."""
    val = getattr(ds, tag, default)
    return default if val is None else val


def _slice_position(ds) -> float:
    """
    Signed distance of this slice along the slice-normal direction.
    Uses ImagePositionPatient projected onto the normal when available,
    falls back to SliceLocation.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None:
        row_dir = np.array([float(v) for v in iop[0:3]])
        col_dir = np.array([float(v) for v in iop[3:6]])
        normal  = np.cross(row_dir, col_dir)
        return float(np.dot([float(v) for v in ipp], normal))
    return float(_safe(ds, "SliceLocation", 0.0))


def _rescale(ds) -> np.ndarray:
    """
    Return the pixel data converted to physical units using
    RescaleSlope / RescaleIntercept (default: slope=1, intercept=0).
    For phase images this typically yields values in radians.
    """
    arr      = ds.pixel_array.astype(np.float32)
    slope    = float(_safe(ds, "RescaleSlope",     1.0))
    intercept = float(_safe(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def _build_affine(sorted_slices: list) -> np.ndarray:
    """
    Build a 4×4 RAS affine matrix from an ordered list of pydicom datasets
    (one dataset per 2-D slice, already sorted by slice position).

    Index convention used here:
        i = column index  (fastest changing in pixel_array.T)
        j = row index
        k = slice index

    The LPS→RAS flip (negate x and y) is applied so nibabel interprets
    the volume correctly.
    """
    ds0  = sorted_slices[0]
    iop  = [float(v) for v in ds0.ImageOrientationPatient]   # 6 values
    ipp0 = np.array([float(v) for v in ds0.ImagePositionPatient])
    ps   = [float(v) for v in ds0.PixelSpacing]              # [row_spacing, col_spacing]

    # row_dir: direction when moving along a row (i.e. column / i direction)
    # col_dir: direction when moving along a column (i.e. row / j direction)
    row_dir = np.array(iop[0:3])
    col_dir = np.array(iop[3:6])

    col_spacing = ps[1]   # spacing between adjacent columns = Δ i
    row_spacing = ps[0]   # spacing between adjacent rows    = Δ j

    # Slice direction + spacing from two consecutive slice origins
    if len(sorted_slices) > 1:
        ipp1      = np.array([float(v) for v in sorted_slices[1].ImagePositionPatient])
        slice_vec = ipp1 - ipp0
        slice_spc = float(np.linalg.norm(slice_vec))
        slice_dir = slice_vec / slice_spc if slice_spc > 0 else np.cross(row_dir, col_dir)
    else:
        slice_dir = np.cross(row_dir, col_dir)
        slice_spc = float(_safe(ds0, "SliceThickness", 1.0))

    # Affine in LPS
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = row_dir * col_spacing    # i direction
    affine_lps[:3, 1] = col_dir * row_spacing    # j direction
    affine_lps[:3, 2] = slice_dir * slice_spc    # k direction
    affine_lps[:3, 3] = ipp0                     # origin

    # LPS → RAS: negate x (L→R) and y (P→A) components
    lps2ras = np.diag([-1., -1., 1., 1.])
    return lps2ras @ affine_lps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_te_from_dicoms(file_paths: list[str]) -> list[float]:
    """
    Fast pass: read only DICOM headers (no pixel data) and return unique
    echo times in **seconds**, sorted ascending.

    Parameters
    ----------
    file_paths : list of str
        Paths to individual DICOM files.

    Returns
    -------
    list of float  – echo times in seconds (e.g. [0.004, 0.008, 0.012]).
    """
    pydicom = _require_pydicom()
    te_set: set[float] = set()
    for p in file_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            et = getattr(ds, "EchoTime", None)
            if et is not None:
                te_set.add(round(float(et) / 1000.0, 9))   # ms → s
        except Exception:
            continue
    return sorted(te_set)


def read_dicom_series(
    file_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Read a single- or multi-echo DICOM series into a numpy volume.

    Parameters
    ----------
    file_paths : list of str
        All DICOM slice files belonging to one series (phase **or** magnitude).
        Multi-echo: pass all echoes together; the function groups them.

    Returns
    -------
    vol : np.ndarray
        Shape ``(H, W, D)`` for single-echo or ``(H, W, D, N)`` for multi-echo.
        Pixel values are rescaled to physical units (radians for phase,
        arbitrary units for magnitude).
    affine : np.ndarray, shape (4, 4)
        RAS affine suitable for a NIfTI header.
    te_values : list of float
        Echo times in seconds, sorted ascending.
    """
    pydicom = _require_pydicom()

    # ── Load all slices ──────────────────────────────────────────────────────
    datasets = []
    for p in file_paths:
        try:
            ds = pydicom.dcmread(p, force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise ValueError(
            "No valid DICOM slices found.  "
            "Make sure you selected all *.dcm files from the series."
        )

    # ── Group by EchoTime ────────────────────────────────────────────────────
    echo_groups: dict[float, list] = {}
    for ds in datasets:
        et_ms = round(float(_safe(ds, "EchoTime", 0.0)), 6)
        echo_groups.setdefault(et_ms, []).append(ds)

    echo_times_ms = sorted(echo_groups.keys())
    te_values_s   = [et / 1000.0 for et in echo_times_ms]

    # ── Validate consistent slice counts ────────────────────────────────────
    n_slices_per_echo = {et: len(v) for et, v in echo_groups.items()}
    counts = list(n_slices_per_echo.values())
    if len(set(counts)) > 1:
        raise ValueError(
            f"Echo groups have inconsistent slice counts: {n_slices_per_echo}.  "
            "Check that all echoes from a single series are uploaded together."
        )
    n_slices = counts[0]

    # ── Build one 3-D volume per echo ────────────────────────────────────────
    affine    = None
    echo_vols = []

    for et_ms in echo_times_ms:
        slices = sorted(echo_groups[et_ms], key=_slice_position)

        # pixel_array is (rows, cols); transpose → (cols, rows) = (i, j)
        stack = np.stack(
            [_rescale(ds).T for ds in slices], axis=2
        ).astype(np.float32)                                    # (H, W, D)

        echo_vols.append(stack)
        if affine is None:
            affine = _build_affine(slices)

    # ── Combine echoes ───────────────────────────────────────────────────────
    if len(echo_vols) == 1:
        vol = echo_vols[0]                                      # (H, W, D)
    else:
        vol = np.stack(echo_vols, axis=3)                       # (H, W, D, N)

    return vol, affine, te_values_s


def dicoms_to_nifti(
    file_paths: list[str],
    out_dir: str | None = None,
    label: str = "input",
) -> tuple[str, list[float]]:
    """
    Convert a DICOM series to a temporary NIfTI file.

    Parameters
    ----------
    file_paths : list of str
    out_dir    : directory in which to write the NIfTI (temp dir if None)
    label      : filename stem (e.g. ``"phase"`` or ``"mag"``)

    Returns
    -------
    nii_path  : str           – absolute path to the saved ``.nii.gz``
    te_values : list[float]   – echo times in seconds
    """
    vol, affine, te_values = read_dicom_series(file_paths)

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="iqsm_dcm_")
    os.makedirs(out_dir, exist_ok=True)

    nii_path = os.path.join(out_dir, f"{label}.nii.gz")
    nib.save(nib.Nifti1Image(vol, affine), nii_path)
    return nii_path, te_values
