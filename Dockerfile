# InvoiceAI - GPU image for the REST API (default) or the Gradio web UI.
#
# Build:  docker build -t invoice-ai .
# Run:    docker run --gpus all -p 8000:8000 -v invoice-ai-models:/models invoice-ai
# UI:     docker run --gpus all -p 7860:7860 -v invoice-ai-models:/models invoice-ai \
#             python -m ui.gradio_app --port 7860
#
# Why python:slim and not an nvidia/cuda base image?
# The PyTorch wheels from pip already contain the CUDA libraries they need.
# A CUDA base image would add a second copy (2-3 GB) for nothing.
# The host only needs an NVIDIA driver and the NVIDIA Container Toolkit.
#
# The model (~7 GB) is NOT baked into the image. It is downloaded on the first
# start into /models; mount a volume there so it is downloaded only once.

FROM python:3.13-slim-bookworm

# PyTorch wheel source. The default (PyPI) gives torch 2.11 with CUDA 13, which
# needs an NVIDIA driver >= 580. For an older driver, use e.g.
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    GRADIO_ANALYTICS_ENABLED=False

# OpenCV (used by RapidOCR) needs these two system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not reinstall them (faster rebuilds).
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url "${TORCH_INDEX_URL}"
COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Load RapidOCR once, so a broken OCR setup (e.g. a missing system library) fails the build, not the first request.
    && python -c "from rapidocr import RapidOCR; RapidOCR()"

# Run as a normal user, not root.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /models \
    && chown appuser:appuser /models
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser api/ api/
COPY --chown=appuser:appuser ui/ ui/
COPY --chown=appuser:appuser samples/ samples/
USER appuser

EXPOSE 8000 7860

# Healthy when the model is loaded. The first start downloads the model, so allow 15 minutes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
