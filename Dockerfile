# ============================================================
# deepMRI iQSM+ – Docker image
# ============================================================
# Build:
#   docker build -t deepmri .
#
# Run (GPU):
#   docker run --gpus all -p 7860:7860 deepmri
#
# Run (CPU only):
#   docker run -p 7860:7860 deepmri
#
# Then open http://localhost:7860 in your browser.
# ============================================================

# Base: PyTorch with CUDA 12.1 (falls back to CPU automatically at runtime)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Hongfu Sun <hongfu.sun@uq.edu.au>"
LABEL description="deepMRI iQSM+ – QSM reconstruction web interface"

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Copy repository ──────────────────────────────────────────────────────────
WORKDIR /deepMRI
COPY . .

# ── Expose Gradio port ───────────────────────────────────────────────────────
EXPOSE 7860

# ── Default command: launch Gradio app ──────────────────────────────────────
CMD ["python", "app/app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
