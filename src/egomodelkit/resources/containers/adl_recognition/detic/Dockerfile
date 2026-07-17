FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG EGOVIZML_REPOSITORY_URL
ARG EGOVIZML_COMMIT_SHA

ARG DETIC_REPOSITORY_URL
ARG DETIC_COMMIT_SHA

ARG DETECTRON2_REPOSITORY_URL
ARG DETECTRON2_COMMIT_SHA

ARG DETIC_WEIGHTS_URL
ARG DETIC_WEIGHTS_FILENAME

ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG PYTORCH_CUDA_INDEX_URL
ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive
ENV EGOVIZML_HOME=/opt/EgoVizML
ENV DETIC_HOME=/opt/Detic
ENV DETECTRON2_HOME=/opt/detectron2

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV EGOMODELKIT_PYTORCH_VERSION=${PYTORCH_VERSION}

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ca-certificates \
    build-essential \
    ninja-build \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade \
    pip \
    setuptools \
    wheel

RUN python3.10 -m pip install --no-cache-dir "gdown==5.2.0"

RUN git clone "${EGOVIZML_REPOSITORY_URL}" "${EGOVIZML_HOME}" \
    && cd "${EGOVIZML_HOME}" \
    && git checkout "${EGOVIZML_COMMIT_SHA}"

RUN git clone "${DETECTRON2_REPOSITORY_URL}" "${DETECTRON2_HOME}" \
    && cd "${DETECTRON2_HOME}" \
    && git checkout "${DETECTRON2_COMMIT_SHA}"

RUN git clone --recurse-submodules "${DETIC_REPOSITORY_URL}" "${DETIC_HOME}" \
    && cd "${DETIC_HOME}" \
    && git checkout "${DETIC_COMMIT_SHA}" \
    && git submodule update --init --recursive

RUN python3.10 -m pip install \
    torch=="${PYTORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}" \
    --index-url "${PYTORCH_CUDA_INDEX_URL}"

WORKDIR ${DETECTRON2_HOME}

RUN MAX_JOBS=2 \
    FORCE_CUDA=1 \
    python3.10 -m pip install \
        --no-build-isolation \
        --verbose \
        -e .

ENV PYTHONPATH="${DETECTRON2_HOME}:${DETIC_HOME}:${EGOVIZML_HOME}"

WORKDIR ${DETIC_HOME}

RUN python3.10 -m pip install -r requirements.txt

# Keep legacy Detic/Detectron2 runtime dependencies compatible after
# Detic's requirements have been installed.
RUN python3.10 -m pip install --no-cache-dir --force-reinstall \
    "numpy==1.26.4" \
    "Pillow==9.5.0"

RUN python3.10 - <<'PY'
import os

import numpy
import PIL
import torch
import torchvision

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("numpy:", numpy.__version__)
print("Pillow:", PIL.__version__)
print("cuda extension targets:", os.environ["TORCH_CUDA_ARCH_LIST"])

if not torch.__version__.startswith(os.environ["EGOMODELKIT_PYTORCH_VERSION"]):
    raise RuntimeError(f"Unexpected torch version: {torch.__version__}")

if torch.version.cuda != "12.8":
    raise RuntimeError(f"Unexpected torch CUDA version: {torch.version.cuda}")

if not numpy.__version__.startswith("1.26."):
    raise RuntimeError(f"Unexpected numpy version: {numpy.__version__}")

if PIL.__version__ != "9.5.0":
    raise RuntimeError(f"Unexpected Pillow version: {PIL.__version__}")
PY

RUN python3.10 -c "\
from detectron2.config import get_cfg; \
from detectron2.layers import nms; \
print('detectron2 import ok')"

RUN mkdir -p models && \
    python3.10 -m gdown --fuzzy "${DETIC_WEIGHTS_URL}" \
    -O "models/${DETIC_WEIGHTS_FILENAME}" && \
    test -s "models/${DETIC_WEIGHTS_FILENAME}"

COPY patch_egovizml_run_detic.py /tmp/patch_egovizml_run_detic.py

RUN python3.10 /tmp/patch_egovizml_run_detic.py \
    "${EGOVIZML_HOME}/scripts/run_detic.py"

ENTRYPOINT ["python3.10", "/opt/EgoVizML/scripts/run_detic.py"]
