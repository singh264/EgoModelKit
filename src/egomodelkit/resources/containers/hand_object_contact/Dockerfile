FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG SHAN_REPOSITORY_URL
ARG SHAN_COMMIT_SHA

ARG CHECKPOINT_GOOGLE_DRIVE_FILE_ID
ARG CHECKPOINT_SESSION
ARG CHECKPOINT_EPOCH
ARG CHECKPOINT_STEP
ARG CHECKPOINT_FILENAME

ARG SHAN_NETWORK_NAME
ARG SHAN_DATASET_NAME
ARG SHAN_MODEL_SUBDIR
ARG SHAN_LOAD_DIR

ENV EGOMODELKIT_SHAN_LOAD_DIR=${SHAN_LOAD_DIR}
ENV EGOMODELKIT_SHAN_NETWORK_NAME=${SHAN_NETWORK_NAME}
ENV EGOMODELKIT_SHAN_DATASET_NAME=${SHAN_DATASET_NAME}
ENV EGOMODELKIT_SHAN_CHECKPOINT_SESSION=${CHECKPOINT_SESSION}
ENV EGOMODELKIT_SHAN_CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH}
ENV EGOMODELKIT_SHAN_CHECKPOINT_STEP=${CHECKPOINT_STEP}

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    build-essential \
    python3.8 \
    python3.8-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    "gdown>=5.2,<6"

RUN python3.8 -m pip install \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN git clone "${SHAN_REPOSITORY_URL}" /opt/hand_object_detector \
    && cd /opt/hand_object_detector \
    && git checkout "${SHAN_COMMIT_SHA}"

WORKDIR /opt/hand_object_detector

RUN python3.8 -m pip install -r requirements.txt

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=7.5

RUN python3.8 - <<'PY'
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

RUN cd lib && FORCE_CUDA=1 python3.8 setup.py build develop

RUN mkdir -p "${SHAN_LOAD_DIR}/${SHAN_MODEL_SUBDIR}" \
    && python3.8 -m gdown \
        --id "${CHECKPOINT_GOOGLE_DRIVE_FILE_ID}" \
        -O "${SHAN_LOAD_DIR}/${SHAN_MODEL_SUBDIR}/${CHECKPOINT_FILENAME}" \
    && test -s "${SHAN_LOAD_DIR}/${SHAN_MODEL_SUBDIR}/${CHECKPOINT_FILENAME}"

COPY entrypoint.py /opt/egomodelkit_hand_object_contact_entrypoint.py

ENTRYPOINT ["python3.8", "/opt/egomodelkit_hand_object_contact_entrypoint.py"]
