FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
# Install CUDA-built torch/vision/audio (matching trio) from PyTorch's index.
# lang-segment-anything requires torch>=2.3.1, so we overwrite the base image's
# 2.3.0 with 2.3.1 cu121 wheels. Without this, pip resolves torch off PyPI and
# pulls a CPU wheel that breaks GPU inference.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Constrain lang-segment-anything's install so it can't drag torch off this pin
# or pull transformers v5 (whose top-level torchaudio import via
# ParakeetForRNNTLoss breaks on older torch ABIs).
RUN printf "torch==2.3.1\ntorchvision==0.18.1\ntorchaudio==2.3.1\ntransformers<5\n" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt \
        "git+https://github.com/luca-medeiros/lang-segment-anything.git" \
        paho-mqtt \
        Pillow \
        numpy

COPY lucid_langsam_server.py .
CMD ["python", "lucid_langsam_server.py"]
