FROM python:3.11-slim

WORKDIR /app

# System libs: build tools for the insightface native build, libglib/libgomp for
# opencv-headless + onnxruntime, curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# On first boot, download the ONNX models into models/ if absent
# (set AUTO_DOWNLOAD_MODELS=false to skip and mount your own).
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
