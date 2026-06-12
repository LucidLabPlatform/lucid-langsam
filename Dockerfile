FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    "git+https://github.com/luca-medeiros/lang-segment-anything.git" \
    paho-mqtt \
    Pillow \
    numpy

# lang-segment-anything pulls in a newer torch (e.g. 2.12) but leaves the base
# image's torchaudio (2.3) in place — libtorchaudio.so then fails to load with
# an undefined-symbol error because its ABI doesn't match the upgraded torch.
# Reinstall torchaudio pinned to whatever torch version actually got installed.
RUN TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
    pip install --no-cache-dir --force-reinstall --no-deps "torchaudio==${TORCH_VERSION}"

COPY lucid_langsam_server.py .
CMD ["python", "lucid_langsam_server.py"]
