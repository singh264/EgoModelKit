FROM python:3.11-slim

ARG EGOVIZML_REPOSITORY_URL
ARG EGOVIZML_COMMIT_SHA

ENV DEBIAN_FRONTEND=noninteractive
ENV EGOVIZML_HOME=/opt/EgoVizML

WORKDIR /opt

RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    ffmpeg \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN git clone "${EGOVIZML_REPOSITORY_URL}" "${EGOVIZML_HOME}" \
    && cd "${EGOVIZML_HOME}" \
    && git checkout "${EGOVIZML_COMMIT_SHA}"

WORKDIR ${EGOVIZML_HOME}

RUN python -m pip install \
    moviepy==1.0.3 \
    opencv-python-headless==4.9.0.80 \
    pandas==2.1.4 \
    polars==0.19.19 \
    scikit-learn==1.3.1 \
    joblib \
    tqdm \
    numpy==1.26.4 \
    pyarrow \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN python -m pip install -e .

COPY entrypoint.py /opt/egomodelkit_adl_entrypoint.py
COPY predict_adl.py /opt/egomodelkit_predict_adl.py

ENTRYPOINT ["python", "/opt/egomodelkit_adl_entrypoint.py"]
