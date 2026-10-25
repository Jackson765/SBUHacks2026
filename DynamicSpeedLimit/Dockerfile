FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000 \
    DATA_DIR=/data CALIBRATION_PATH=/data/calibration.json CALIBRATION_TEMP_PATH=/data/calibrationtemp.json
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/requirements.txt
# The CPU wheel avoids downloading unused CUDA libraries for a CPU deployment.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY frontend frontend
COPY YOLO YOLO
COPY model_b/predict.py model_b/predict.py
COPY model_b/models model_b/models
COPY traffic traffic
RUN useradd --create-home --uid 10001 flow && mkdir -p /data && chown flow:flow /data
USER flow
EXPOSE 8000
CMD ["python", "-m", "backend.app"]
