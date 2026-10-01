# NVIDIA GPU runtime. Apple Silicon should run broadcast.py on the host, not in this image.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY broadcast.py .

# Webcam and virtual camera need the host devices passed in.
# docker run --gpus all --device /dev/video0 -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gpu-broadcast
CMD ["python", "broadcast.py", "--width", "1280", "--height", "720"]
