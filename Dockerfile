FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
RUN pip install --no-cache-dir \
    lang-segment-anything \
    paho-mqtt \
    opencv-python-headless \
    Pillow \
    numpy

COPY lucid_langsam_server.py .
CMD ["python", "lucid_langsam_server.py"]
