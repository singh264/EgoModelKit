FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG SHAN_REPOSITORY_URL
ARG SHAN_COMMIT_SHA

ARG CHECKPOINT_SOURCE_URL
ARG CHECKPOINT_FILENAME
ARG CHECKPOINT_SESSION
ARG CHECKPOINT_EPOCH
ARG CHECKPOINT_STEP

ARG SHAN_NETWORK_NAME
ARG SHAN_DATASET_NAME
ARG SHAN_MODEL_SUBDIR
ARG SHAN_LOAD_DIR

ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG PYTORCH_CUDA_INDEX_URL
ARG TORCH_CUDA_ARCH_LIST

ENV EGOMODELKIT_SHAN_LOAD_DIR=${SHAN_LOAD_DIR}
ENV EGOMODELKIT_SHAN_NETWORK_NAME=${SHAN_NETWORK_NAME}
ENV EGOMODELKIT_SHAN_DATASET_NAME=${SHAN_DATASET_NAME}
ENV EGOMODELKIT_SHAN_CHECKPOINT_SESSION=${CHECKPOINT_SESSION}
ENV EGOMODELKIT_SHAN_CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH}
ENV EGOMODELKIT_SHAN_CHECKPOINT_STEP=${CHECKPOINT_STEP}

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV EGOMODELKIT_PYTORCH_VERSION=${PYTORCH_VERSION}

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    build-essential \
    ninja-build \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    "gdown>=5.2,<6"

RUN python3.10 -m pip install \
    torch=="${PYTORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}" \
    --index-url "${PYTORCH_CUDA_INDEX_URL}"

RUN git clone "${SHAN_REPOSITORY_URL}" /opt/hand_object_detector \
    && cd /opt/hand_object_detector \
    && git checkout "${SHAN_COMMIT_SHA}"

WORKDIR /opt/hand_object_detector

RUN python3.10 -m pip install -r requirements.txt

RUN python3.10 -m pip install --no-cache-dir --force-reinstall \
    "numpy==1.26.4"

RUN python3.10 - <<'PY'
import os

import numpy
import torch
import torchvision

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("numpy:", numpy.__version__)
print("cuda extension targets:", os.environ["TORCH_CUDA_ARCH_LIST"])

if not torch.__version__.startswith(os.environ["EGOMODELKIT_PYTORCH_VERSION"]):
    raise RuntimeError(f"Unexpected torch version: {torch.__version__}")

if torch.version.cuda != "12.8":
    raise RuntimeError(f"Unexpected torch CUDA version: {torch.version.cuda}")

if not numpy.__version__.startswith("1.26."):
    raise RuntimeError(f"Unexpected numpy version: {numpy.__version__}")
PY

RUN python3.10 - <<'PY'
from pathlib import Path

setup_py = Path("/opt/hand_object_detector/lib/setup.py")
text = setup_py.read_text()

old = "if torch.cuda.is_available() and CUDA_HOME is not None:"
new = (
    'if (torch.cuda.is_available() or os.environ.get("FORCE_CUDA") == "1") '
    "and CUDA_HOME is not None:"
)

if old not in text:
    raise RuntimeError("Expected Shan CUDA build guard was not found in lib/setup.py")

setup_py.write_text(text.replace(old, new))
PY

RUN python3.10 - <<'PY'
from pathlib import Path

replacements = {
    "model/csrc/cpu/ROIAlign_cpu.cpp": {
        "AT_DISPATCH_FLOATING_TYPES(input.type(),": (
            "AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),"
        ),
    },
    "model/csrc/cpu/nms_cpu.cpp": {
        "AT_DISPATCH_FLOATING_TYPES(dets.type(),": (
            "AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(),"
        ),
    },
    "model/csrc/cuda/ROIPool_cuda.cu": {
        "AT_DISPATCH_FLOATING_TYPES(input.type(),": (
            "AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),"
        ),
        "AT_DISPATCH_FLOATING_TYPES(grad.type(),": (
            "AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),"
        ),
    },
    "model/csrc/cuda/ROIAlign_cuda.cu": {
        "AT_DISPATCH_FLOATING_TYPES(input.type(),": (
            "AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),"
        ),
        "AT_DISPATCH_FLOATING_TYPES(grad.type(),": (
            "AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),"
        ),
    },
}

base = Path("/opt/hand_object_detector/lib")

for relative_path, file_replacements in replacements.items():
    path = base / relative_path
    text = path.read_text()

    for old, new in file_replacements.items():
        if old not in text:
            raise RuntimeError(
                f"Expected pattern not found in {relative_path}: {old}"
            )

        text = text.replace(old, new)

    path.write_text(text)
PY

RUN cd lib && \
    MAX_JOBS=2 \
    FORCE_CUDA=1 \
    python3.10 setup.py build develop

RUN mkdir -p "${SHAN_LOAD_DIR}/${SHAN_MODEL_SUBDIR}" \
    && python3.10 -m gdown \
        --fuzzy "${CHECKPOINT_SOURCE_URL}" \
        -O "${SHAN_LOAD_DIR}/${SHAN_MODEL_SUBDIR}/${CHECKPOINT_FILENAME}" \
    && test -s "${SHAN_LOAD_DIR}/${SHAN_MODEL_SUBDIR}/${CHECKPOINT_FILENAME}"

COPY entrypoint.py /opt/egomodelkit_hand_object_contact_entrypoint.py

ENTRYPOINT ["python3.10", "/opt/egomodelkit_hand_object_contact_entrypoint.py"]
