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

# lang-segment-anything's deps pull in torch 2.12 (bleeding edge — its matching
# torchaudio hasn't shipped on PyPI yet) and leave the base image's torchaudio
# 2.3 in place, so libtorchaudio.so fails to load with an undefined-symbol error.
# Force-reinstall a known-matching torch+torchaudio pair (2.11.0 is the latest
# torchaudio published on PyPI as of writing). Pip will downgrade torch to match.
RUN pip install --no-cache-dir "torch==2.11.0" "torchaudio==2.11.0"

COPY lucid_langsam_server.py .
CMD ["python", "lucid_langsam_server.py"]
