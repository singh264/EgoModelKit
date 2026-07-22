FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.py /opt/egomodelkit_hand_interaction_entrypoint.py

ENTRYPOINT ["python", "/opt/egomodelkit_hand_interaction_entrypoint.py"]
