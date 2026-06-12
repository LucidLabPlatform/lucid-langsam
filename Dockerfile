FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    "git+https://github.com/luca-medeiros/lang-segment-anything.git" \
    paho-mqtt \
    opencv-python-headless \
    Pillow \
    numpy

COPY lucid_langsam_server.py .
CMD ["python", "lucid_langsam_server.py"]
