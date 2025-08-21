FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir "scikit-learn==1.2.2"

COPY app.py .
COPY pm25_predictor.py .

COPY models/ ./models/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS=0

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]