FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
# Constrain lang-segment-anything's pip install so it can't drag torch (or its
# ABI-tied siblings torchvision/torchaudio) off the base image's CUDA-built
# versions. Without this, pip pulls in a bleeding-edge CPU torch 2.12, leaving
# the original torchaudio/torchvision in place with mismatched ABIs.
RUN printf "torch==2.3.0\ntorchvision==0.18.0\ntorchaudio==2.3.0\n" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt \
        "git+https://github.com/luca-medeiros/lang-segment-anything.git" \
        paho-mqtt \
        Pillow \
        numpy

COPY lucid_langsam_server.py .
CMD ["python", "lucid_langsam_server.py"]
