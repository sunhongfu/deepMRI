"""
deepMRI iQSM+ – Gradio Web Interface
=====================================
Clinician-friendly web UI for Quantitative Susceptibility Mapping (QSM).

Launch:
    python app/app.py                   # CPU
    python app/app.py --share           # public Gradio link
    python app/app.py --server-port 8080

Docker:
    docker compose up                   # see docker-compose.yml at repo root
"""

import argparse
import os
import tempfile
import traceback

import gradio as gr
import nibabel as nib
import numpy as np

from inference import run_iqsm_plus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_floats(text: str, name: str, n: int | None = None) -> list[float]:
    """Parse a comma- or space-separated string of floats."""
    try:
        vals = [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        raise gr.Error(f"'{name}' must be numbers separated by spaces or commas.")
    if n is not None and len(vals) != n:
        raise gr.Error(f"'{name}' must have exactly {n} values, got {len(vals)}.")
    return vals


def _make_slice_figure(nii_path: str):
    """
    Return (axial_img, coronal_img, sagittal_img) as numpy uint8 arrays
    from the middle slice of a NIfTI volume.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = nib.load(nii_path).get_fdata(dtype=np.float32)
    # Normalise to [0, 1] for display
    vmin, vmax = np.percentile(vol, [2, 98])
    vol_n = np.clip((vol - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    slices = {
        "Axial":    vol_n[:, :, vol_n.shape[2] // 2].T,
        "Coronal":  vol_n[:, vol_n.shape[1] // 2, :].T,
        "Sagittal": vol_n[vol_n.shape[0] // 2, :, :].T,
    }

    imgs = []
    for title, sl in slices.items():
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        # Render to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imgs.append(buf)
        plt.close(fig)

    return imgs[0], imgs[1], imgs[2]


# ---------------------------------------------------------------------------
# Core reconstruction callback
# ---------------------------------------------------------------------------

def reconstruct(
    phase_file,
    te_str,
    mag_file,
    mask_file,
    voxel_str,
    b0dir_str,
    b0_val,
    eroded_rad,
    progress=gr.Progress(track_tqdm=True),
):
    # ── Validate compulsory inputs ──────────────────────────────────────────
    if phase_file is None:
        raise gr.Error("Please upload a phase NIfTI file.")
    if not te_str.strip():
        raise gr.Error("Please enter at least one echo time (TE).")

    te_values = _parse_floats(te_str, "Echo time(s) (TE)")
    if any(t <= 0 for t in te_values):
        raise gr.Error("Echo times must be positive. Enter values in seconds (e.g. 0.020).")

    # ── Optional parameters ─────────────────────────────────────────────────
    voxel_size = None
    if voxel_str.strip():
        voxel_size = _parse_floats(voxel_str, "Voxel size", n=3)
        if any(v <= 0 for v in voxel_size):
            raise gr.Error("Voxel sizes must be positive.")

    b0_dir = None
    if b0dir_str.strip():
        b0_dir = _parse_floats(b0dir_str, "B0 direction", n=3)
        if np.linalg.norm(b0_dir) == 0:
            raise gr.Error("B0 direction must not be the zero vector.")

    # ── Run reconstruction ──────────────────────────────────────────────────
    output_dir = tempfile.mkdtemp(prefix="iqsm_plus_out_")

    def _progress(frac, msg):
        progress(frac, desc=msg)

    try:
        out_path = run_iqsm_plus(
            phase_nii_path=phase_file.name,
            te_values=te_values,
            mag_nii_path=mag_file.name if mag_file else None,
            mask_nii_path=mask_file.name if mask_file else None,
            voxel_size=voxel_size,
            b0_dir=b0_dir,
            b0=float(b0_val),
            eroded_rad=int(eroded_rad),
            output_dir=output_dir,
            progress_fn=_progress,
        )
    except Exception:
        raise gr.Error(
            "Reconstruction failed. Check the log for details.\n\n"
            + traceback.format_exc()
        )

    # ── Visualise result ────────────────────────────────────────────────────
    try:
        ax_img, cor_img, sag_img = _make_slice_figure(out_path)
    except Exception:
        ax_img = cor_img = sag_img = None

    status = (
        "✅ Reconstruction complete!\n"
        f"Output saved to: {out_path}\n\n"
        "Download the NIfTI file below and open it in your preferred viewer "
        "(e.g. FSLeyes, ITK-SNAP, 3D Slicer)."
    )

    return status, out_path, ax_img, cor_img, sag_img


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

TITLE = "deepMRI – iQSM+ QSM Reconstruction"
DESCRIPTION = """
**Quantitative Susceptibility Mapping (QSM)** from MRI phase data
using the *iQSM+* deep learning model ([paper](https://doi.org/10.1016/j.media.2024.103160)).

**Quick-start:**
1. Upload your **phase** NIfTI file (`.nii` or `.nii.gz`).
2. Enter the echo time(s) **in seconds** (comma-separated for multi-echo).
3. Adjust parameters as needed and click **Run Reconstruction**.
4. Download the QSM result and open it in FSLeyes / ITK-SNAP / 3D Slicer.
"""

HELP_TE = (
    "Echo time(s) in **seconds**. "
    "Single-echo example: `0.020`. "
    "Multi-echo example: `0.004, 0.008, 0.012, 0.016, 0.020`."
)
HELP_VOX = (
    "Voxel size in mm (x y z). "
    "Leave blank to read from the NIfTI header (recommended). "
    "Example: `1 1 2`."
)
HELP_B0DIR = (
    "B0 field direction vector (unit vector). "
    "Leave blank for default `0 0 1` (pure axial). "
    "Example for tilted acquisition: `0.04 0.05 0.998`."
)
HELP_PHASE = (
    "Wrapped phase NIfTI file. "
    "Expects **phase = −ΔB · γ · TE** convention. "
    "3D (single-echo) or 4D (multi-echo) volumes are both supported."
)


def build_ui():
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Required inputs")

                phase_file = gr.File(
                    label="Phase NIfTI (.nii / .nii.gz)",
                    file_types=[".nii", ".gz"],
                )
                gr.Markdown(f"<small>{HELP_PHASE}</small>")

                te_str = gr.Textbox(
                    label="Echo time(s) – TE (seconds)",
                    placeholder="e.g.  0.020   or   0.004, 0.008, 0.012",
                )
                gr.Markdown(f"<small>{HELP_TE}</small>")

                gr.Markdown("### Optional inputs")

                mag_file = gr.File(
                    label="Magnitude NIfTI (optional)",
                    file_types=[".nii", ".gz"],
                )
                mask_file = gr.File(
                    label="Brain mask NIfTI (optional – 3D binary)",
                    file_types=[".nii", ".gz"],
                )

                gr.Markdown("### Acquisition parameters")

                with gr.Row():
                    b0_val = gr.Number(
                        label="B0 field strength (Tesla)",
                        value=3.0,
                        minimum=0.1,
                        maximum=14.0,
                        step=0.5,
                    )
                    eroded_rad = gr.Slider(
                        label="Mask erosion radius (voxels)",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=3,
                    )

                voxel_str = gr.Textbox(
                    label="Voxel size override (mm, optional)",
                    placeholder="e.g.  1 1 2",
                )
                gr.Markdown(f"<small>{HELP_VOX}</small>")

                b0dir_str = gr.Textbox(
                    label="B0 direction override (optional)",
                    placeholder="e.g.  0 0 1",
                )
                gr.Markdown(f"<small>{HELP_B0DIR}</small>")

                run_btn = gr.Button("▶ Run Reconstruction", variant="primary", size="lg")

            # ── Right column: outputs ────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Results")

                status_box = gr.Textbox(
                    label="Status",
                    lines=4,
                    interactive=False,
                    placeholder="Reconstruction output will appear here …",
                )
                download_btn = gr.File(label="Download QSM NIfTI", interactive=False)

                gr.Markdown("#### Preview (middle slice)")
                with gr.Row():
                    axial_img   = gr.Image(label="Axial",    show_label=True)
                    coronal_img = gr.Image(label="Coronal",  show_label=True)
                    sagittal_img = gr.Image(label="Sagittal", show_label=True)

        # ── Wire up the button ───────────────────────────────────────────
        run_btn.click(
            fn=reconstruct,
            inputs=[
                phase_file, te_str,
                mag_file, mask_file,
                voxel_str, b0dir_str,
                b0_val, eroded_rad,
            ],
            outputs=[status_box, download_btn, axial_img, coronal_img, sagittal_img],
        )

        gr.Markdown(
            "---\n"
            "**Citation:** Gao Y, et al. *Plug-and-Play latent feature editing for "
            "orientation-adaptive QSM neural networks.* "
            "Medical Image Analysis, 2024. "
            "[doi:10.1016/j.media.2024.103160](https://doi.org/10.1016/j.media.2024.103160)\n\n"
            "**Source code:** [github.com/sunhongfu/deepMRI](https://github.com/sunhongfu/deepMRI)"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="deepMRI iQSM+ Gradio server")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
    )
