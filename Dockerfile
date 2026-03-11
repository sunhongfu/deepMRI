# ============================================================
# deepMRI iQSM+ – Docker image  (multi-platform)
# ============================================================
# Supports:
#   Platform              | Auto-detected | Notes
#   ----------------------|---------------|------------------------
#   Apple Silicon (M1–M4) | arm64         | CPU inference
#   Intel/AMD Mac         | amd64         | CPU inference
#   Windows (Docker WSL2) | amd64         | CPU inference
#   Linux AMD64, no GPU   | amd64         | CPU inference
#   Linux AMD64, NVIDIA   | amd64         | GPU – set TORCH_VARIANT
#
# Default build (auto-detects platform, CPU inference):
#   docker compose build
#   docker compose up
#
# NVIDIA GPU build (Linux + NVIDIA Container Toolkit only):
#   TORCH_VARIANT=cu121 docker compose build
#   docker compose up   ← also uncomment the GPU block in docker-compose.yml
# ============================================================

FROM python:3.10-slim

LABEL maintainer="Hongfu Sun <hongfu.sun@uq.edu.au>"
LABEL description="deepMRI iQSM+ – QSM reconstruction web interface"

# Docker injects this automatically at build time: arm64 | amd64
ARG TARGETARCH

# PyTorch index to use.
#   cpu   → CPU-only wheels (default, works everywhere without a GPU)
#   cu121 → CUDA 12.1 wheels (NVIDIA GPU on Linux)
#   cu118 → CUDA 11.8 wheels (older NVIDIA GPU on Linux)
ARG TORCH_VARIANT=cpu

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch  ─────────────────────────────────────────────────────────────────
# arm64  → wheels come from PyPI directly (native ARM build, no CUDA index needed)
# amd64  → wheels come from PyTorch's index; TORCH_VARIANT selects cpu / cu121 / cu118
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir "torch>=2.1.0"; \
    else \
        pip install --no-cache-dir "torch>=2.1.0" \
            --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"; \
    fi

# ── Other Python dependencies ─────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Copy repository ──────────────────────────────────────────────────────────
WORKDIR /deepMRI
COPY . .

# ── Expose Gradio port ───────────────────────────────────────────────────────
EXPOSE 7860

# ── Launch ───────────────────────────────────────────────────────────────────
CMD ["python", "app/app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
