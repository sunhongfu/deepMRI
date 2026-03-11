"""
Pure Python inference pipeline for iQSM+.

Replaces the MATLAB preprocessing/postprocessing steps with numpy/scipy
so the tool runs without any MATLAB dependency.
"""

import os
import sys
import tempfile

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy.ndimage import zoom, binary_erosion

# ---------------------------------------------------------------------------
# Locate model code and checkpoints
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEEPMRI_DIR = os.path.dirname(_HERE)
_IQSM_PLUS_DIR = os.path.join(_DEEPMRI_DIR, "iQSM_Plus")
_INFERENCE_DIR = os.path.join(
    _IQSM_PLUS_DIR,
    "PythonCodes", "Evaluation", "iQSM_series", "iQSM_plus_v1",
)
CHECKPOINTS_DIR = os.path.join(
    _IQSM_PLUS_DIR,
    "PythonCodes", "Evaluation", "checkpoints", "iQSM_plus_v1",
)

if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

from LoT_Unet_plus import LoT_Unet, LoTLayer  # noqa: E402 (after sys.path edit)
from Unet import Unet  # noqa: E402


# ---------------------------------------------------------------------------
# Laplacian kernel (matches the hardcoded kernel in Inference_iQSMSeries.py)
# ---------------------------------------------------------------------------
_CONV_OP = np.array(
    [
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
        [[3 / 26, 3 / 13, 3 / 26], [3 / 13, -44 / 13, 3 / 13], [3 / 26, 3 / 13, 3 / 26]],
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Model loading (cached globally so the Gradio app doesn't reload per call)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def get_model(device: torch.device) -> nn.Module:
    """Load (or return cached) iQSM+ model."""
    key = str(device)
    if key in _model_cache:
        return _model_cache[key]

    conv_op = torch.from_numpy(_CONV_OP).unsqueeze(0).unsqueeze(0)

    lot_layer = LoTLayer(conv_op)
    lot_layer = nn.DataParallel(lot_layer)
    lot_layer.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINTS_DIR, "LoTLayer_chi.pth"),
            map_location=device,
        )
    )
    lot_layer = lot_layer.module
    lot_layer.eval()

    unet = Unet(4, 16, 1)
    unet = nn.DataParallel(unet)
    unet.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINTS_DIR, "iQSM_plus.pth"),
            map_location=device,
        )
    )
    unet = unet.module
    unet.eval()

    model = LoT_Unet(lot_layer, unet)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    _model_cache[key] = model
    return model


# ---------------------------------------------------------------------------
# Preprocessing helpers  (pure Python equivalents of MATLAB steps)
# ---------------------------------------------------------------------------

def _make_sphere(radius: int) -> np.ndarray:
    """Spherical binary structuring element of given radius."""
    c = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(c, c, c, indexing='ij')
    return (x**2 + y**2 + z**2) <= radius**2


def _zero_pad(arr: np.ndarray, multiple: int = 16):
    """
    Pad the first three spatial dimensions so each is a multiple of `multiple`.
    Returns (padded_array, positions) where positions = [(x1,x2), (y1,y2), (z1,z2)].
    """
    shape = arr.shape
    pad_spec = []
    positions = []
    for s in shape[:3]:
        total = (multiple - s % multiple) % multiple
        before = total // 2
        after = total - before
        pad_spec.append((before, after))
        positions.append((before, before + s))
    if arr.ndim == 4:
        pad_spec.append((0, 0))
    return np.pad(arr, pad_spec), positions


def _zero_remove(arr: np.ndarray, positions: list) -> np.ndarray:
    """Undo _zero_pad using the saved positions."""
    (x1, x2), (y1, y2), (z1, z2) = positions
    if arr.ndim == 3:
        return arr[x1:x2, y1:y2, z1:z2]
    return arr[x1:x2, y1:y2, z1:z2, :]


def _brain_bbox(mask: np.ndarray, pad: int = 16) -> tuple:
    """
    Bounding box of nonzero mask region, expanded by `pad` voxels on each side
    and clamped to valid array indices.  Returns a tuple of slices (one per dim).
    Falls back to the full volume if the mask is empty.
    """
    nonzero = np.argwhere(mask > 0.5)
    if len(nonzero) == 0:
        return tuple(slice(0, s) for s in mask.shape)
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0)
    return tuple(
        slice(max(0, int(lo) - pad), min(int(s), int(hi) + 1 + pad))
        for lo, hi, s in zip(mins, maxs, mask.shape)
    )


def _interpolate_phase_to_isotropic(phase: np.ndarray, vox: np.ndarray) -> np.ndarray:
    """
    Interpolate phase data to isotropic resolution.
    Uses complex-domain interpolation to preserve phase wraps
    (equivalent to MATLAB: angle(imresize3(exp(1j*phase), new_size))).
    """
    min_vox = vox.min()
    factors = (vox / min_vox).tolist()  # zoom factors per spatial dim
    if phase.ndim == 4:
        factors = factors + [1.0]  # don't zoom along echo dim

    cplx = np.exp(1j * phase)
    real_z = zoom(cplx.real, factors, order=1)
    imag_z = zoom(cplx.imag, factors, order=1)
    return np.angle(real_z + 1j * imag_z).astype(np.float32)


def _interpolate_volume(vol: np.ndarray, vox: np.ndarray) -> np.ndarray:
    """Linear interpolation of a non-phase volume (magnitude, mask)."""
    min_vox = vox.min()
    factors = (vox / min_vox).tolist()
    if vol.ndim == 4:
        factors = factors + [1.0]
    return zoom(vol.astype(np.float32), factors, order=1)


# ---------------------------------------------------------------------------
# Main reconstruction function
# ---------------------------------------------------------------------------

def run_iqsm_plus(
    phase_nii_path: str,
    te_values: list,
    *,
    mag_nii_path: str | None = None,
    mask_nii_path: str | None = None,
    voxel_size: list | None = None,
    b0_dir: list | None = None,
    b0: float = 3.0,
    eroded_rad: int = 3,
    output_dir: str | None = None,
    progress_fn=None,
) -> str:
    """
    Run iQSM+ QSM reconstruction in pure Python.

    Parameters
    ----------
    phase_nii_path : str
        Path to wrapped-phase NIfTI file (3D single-echo or 4D multi-echo).
        Phase convention: phase = -delta_B * gamma * TE.
        If your data uses the opposite sign, negate before calling.
    te_values : list of float
        Echo time(s) in **seconds**, e.g. [0.020] or [0.004, 0.008, …].
    mag_nii_path : str, optional
        Path to magnitude NIfTI (same spatial dims as phase).
    mask_nii_path : str, optional
        Path to brain-mask NIfTI (3D). If not provided, whole volume is used.
    voxel_size : [x, y, z] mm, optional
        Overrides the voxel size from the NIfTI header.
    b0_dir : [x, y, z], optional
        B0 field direction unit vector. Default: [0, 0, 1] (axial).
    b0 : float
        B0 field strength in Tesla. Default: 3.0.
    eroded_rad : int
        Radius (voxels) for brain-mask erosion. Default: 3.
    output_dir : str, optional
        Directory where output NIfTI is written. Defaults to a temp dir.
    progress_fn : callable(float, str), optional
        Called with (fraction_done, message) to report progress.

    Returns
    -------
    str
        Absolute path to the output QSM NIfTI file.
    """

    def _log(frac, msg):
        print(f"[{frac:.0%}] {msg}")
        if progress_fn is not None:
            progress_fn(frac, msg)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="iqsm_plus_")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(0.0, f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load phase data
    # ------------------------------------------------------------------
    _log(0.05, "Loading phase NIfTI …")
    phase_img = nib.load(phase_nii_path)
    phase = phase_img.get_fdata(dtype=np.float32)
    affine = phase_img.affine

    # Voxel size from header (or user override)
    if voxel_size is not None:
        vox = np.array(voxel_size, dtype=np.float64)
    else:
        zooms = phase_img.header.get_zooms()
        vox = np.array(zooms[:3], dtype=np.float64)

    # B0 direction default
    if b0_dir is None:
        b0_dir = [0.0, 0.0, 1.0]
    b0_dir = np.array(b0_dir, dtype=np.float64)
    b0_dir = b0_dir / np.linalg.norm(b0_dir)

    # TE as numpy array column vector (matches MATLAB TE')
    te = np.array(te_values, dtype=np.float32).reshape(-1)

    # Ensure phase is single-precision
    phase = phase.astype(np.float32)

    # Handle single-echo: add echo dim for uniform handling
    single_echo = phase.ndim == 3
    if single_echo:
        phase = phase[:, :, :, np.newaxis]

    imsize_orig = np.array(phase.shape[:3], dtype=int)  # (H, W, D)
    n_echoes = phase.shape[3]

    # ------------------------------------------------------------------
    # 2. Load magnitude and mask (optional)
    # ------------------------------------------------------------------
    if mag_nii_path is not None:
        _log(0.08, "Loading magnitude NIfTI …")
        mag = nib.load(mag_nii_path).get_fdata(dtype=np.float32)
        if mag.ndim == 3:
            mag = mag[:, :, :, np.newaxis]
    else:
        mag = np.ones_like(phase)

    if mask_nii_path is not None:
        _log(0.10, "Loading brain mask NIfTI …")
        mask = nib.load(mask_nii_path).get_fdata(dtype=np.float32)
    else:
        mask = np.ones(imsize_orig, dtype=np.float32)
        eroded_rad = 0  # no erosion when using whole-head mask

    # ------------------------------------------------------------------
    # 3. Preprocessing (mirrors iQSM_plus.m steps)
    # ------------------------------------------------------------------

    # 3a. Phase sign convention flip (matches MATLAB sf = -1)
    phase = -1.0 * phase

    # 3b. Isotropic interpolation
    interp_flag = not np.allclose(vox, vox.min())
    if interp_flag:
        _log(0.15, "Interpolating to isotropic resolution …")
        phase = _interpolate_phase_to_isotropic(phase, vox)
        mag = _interpolate_volume(mag, vox)
        mask = _interpolate_volume(mask, vox)
        vox_iso = np.full(3, vox.min())
        imsize_iso = np.array(phase.shape[:3], dtype=int)
    else:
        vox_iso = vox.copy()
        imsize_iso = imsize_orig.copy()

    # 3c. Brain-mask erosion
    if eroded_rad > 0:
        _log(0.18, f"Eroding brain mask (radius={eroded_rad}) …")
        struct = _make_sphere(eroded_rad)
        mask_bin = mask > 0.5
        mask = binary_erosion(mask_bin, structure=struct).astype(np.float32)

    # 3d. Dimension permutation so B0 is closest to z-axis
    permute_flag = abs(b0_dir[1]) > abs(b0_dir[2])
    if permute_flag:
        b0_dir[[1, 2]] = b0_dir[[2, 1]]
        phase = np.transpose(phase, (0, 2, 1, 3))
        mag = np.transpose(mag, (0, 2, 1, 3))
        mask = np.transpose(mask, (1, 0, 2))  # mask is 3D

    # 3e. Crop to brain bounding box (+ 16-voxel context padding)
    bbox = _brain_bbox(mask, pad=16)
    phase_crop = phase[bbox + (slice(None),)]   # (H_c, W_c, D_c, N)
    mask_crop  = mask[bbox]                     # (H_c, W_c, D_c)
    _log(0.22, f"Cropped volume: {phase.shape[:3]} → {phase_crop.shape[:3]}")

    # 3f. Zero-pad cropped volume to multiples of 16
    phase_pad, positions = _zero_pad(phase_crop, 16)
    mask_pad, _          = _zero_pad(mask_crop,  16)

    # ------------------------------------------------------------------
    # 4. Deep learning inference
    # ------------------------------------------------------------------
    _log(0.25, "Loading iQSM+ model …")
    model = get_model(device)

    _log(0.30, "Running reconstruction …")

    te_t = torch.from_numpy(te).float().to(device)
    b0_t = torch.tensor([b0], dtype=torch.float32).to(device)
    z_prjs_t = torch.from_numpy(b0_dir.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3)

    # phase_pad shape: (H, W, D, N_echoes)
    phase_t = torch.from_numpy(phase_pad).float()           # (H, W, D, N)
    phase_t = phase_t.permute(3, 0, 1, 2).unsqueeze(1)     # (N, 1, H, W, D)
    phase_t = phase_t.to(device)

    mask_t = torch.from_numpy(mask_pad).float()
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, H, W, D)

    pred_chi = torch.zeros_like(phase_t)  # (N, 1, H, W, D)

    with torch.inference_mode():
        for i in range(n_echoes):
            _log(
                0.30 + 0.50 * (i / n_echoes),
                f"Reconstructing echo {i + 1}/{n_echoes} …",
            )
            tmp_img = phase_t[i].unsqueeze(0)   # (1, 1, H, W, D)
            tmp_te = te_t[i]
            pred_chi[i] = model(tmp_img, mask_t, tmp_te, b0_t, z_prjs_t)

    pred_chi = pred_chi * mask_t                            # apply mask
    pred_chi = pred_chi.squeeze(1)                         # (N, H, W, D)
    pred_chi = pred_chi.permute(1, 2, 3, 0)               # (H, W, D, N)
    pred_chi = pred_chi.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Post-processing
    # ------------------------------------------------------------------
    _log(0.82, "Post-processing …")

    # 5a. Remove zero-padding from cropped result, paste back into full volume
    chi_crop = _zero_remove(pred_chi, positions)           # (H_c, W_c, D_c, N)
    chi = np.zeros((*phase.shape[:3], n_echoes), dtype=np.float32)
    chi[bbox + (slice(None),)] = chi_crop                  # (H, W, D, N)

    # 5b. Multi-echo magnitude+TE-weighted fitting → single QSM map
    # mag is already at full spatial size (H, W, D, N) — no trimming needed
    if n_echoes > 1:
        te_bc = te.reshape(1, 1, 1, -1)
        mag_echo = mag if mag.ndim == 4 else mag[:, :, :, np.newaxis]
        weights = (mag_echo * te_bc) ** 2
        denom = weights.sum(axis=3, keepdims=True)
        denom[denom == 0] = 1.0
        chi_fitted = (weights * chi).sum(axis=3) / denom.squeeze(3)
        chi_fitted = np.nan_to_num(chi_fitted)
    else:
        chi_fitted = chi[:, :, :, 0]

    # 5c. Undo dimension permutation
    if permute_flag:
        chi_fitted = np.transpose(chi_fitted, (0, 2, 1))

    # 5d. Undo isotropic interpolation (back to original resolution)
    if interp_flag:
        factors = (imsize_orig / imsize_iso).tolist()
        chi_fitted = zoom(chi_fitted, factors, order=1)

    # ------------------------------------------------------------------
    # 6. Save output NIfTI
    # ------------------------------------------------------------------
    _log(0.95, "Saving output NIfTI …")
    out_path = os.path.join(output_dir, "iQSM_plus.nii.gz")
    out_img = nib.Nifti1Image(chi_fitted, affine)
    # Store voxel size from original header
    out_img.header.set_zooms(tuple(vox_iso if not interp_flag else vox))
    nib.save(out_img, out_path)

    _log(1.0, f"Done! Saved to {out_path}")
    return out_path
