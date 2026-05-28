FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

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

ENV DEBIAN_FRONTEND=noninteractive
ENV EGOVIZML_HOME=/opt/EgoVizML
ENV DETIC_HOME=/opt/Detic
ENV DETECTRON2_HOME=/opt/detectron2

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
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

RUN python3.8 -m pip install --upgrade pip setuptools wheel

RUN python3.8 -m pip install --no-cache-dir "gdown==5.2.0"

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

RUN python3.8 -m pip install \
    torch=="${PYTORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}" \
    --extra-index-url "${PYTORCH_CUDA_INDEX_URL}"

WORKDIR ${DETECTRON2_HOME}

RUN python3.8 -m pip install --no-build-isolation -e .

ENV PYTHONPATH="${DETECTRON2_HOME}:${DETIC_HOME}:${EGOVIZML_HOME}"

WORKDIR ${DETIC_HOME}

RUN python3.8 -m pip install -r requirements.txt

# Older Detectron2/Detic paths still expect PIL.Image.LINEAR
RUN python3.8 -m pip install --no-cache-dir --force-reinstall "Pillow==9.5.0"

RUN python3.8 -c "\
from detectron2.config import get_cfg; \
from detectron2.layers import nms; \
print('detectron2 import ok')"

RUN mkdir -p models && \
    gdown --fuzzy "${DETIC_WEIGHTS_URL}" \
    -O "models/${DETIC_WEIGHTS_FILENAME}" && \
    test -s "models/${DETIC_WEIGHTS_FILENAME}"

COPY patch_egovizml_run_detic.py /tmp/patch_egovizml_run_detic.py

RUN python3.8 /tmp/patch_egovizml_run_detic.py \
    "${EGOVIZML_HOME}/scripts/run_detic.py"

ENTRYPOINT ["python3.8", "/opt/EgoVizML/scripts/run_detic.py"]
