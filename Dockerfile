
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/v2 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    flask==3.1.0 \
    transformers==4.48.3 \
    huggingface-hub==0.36.2 \
    safetensors \
    && find /usr/local/lib/python3.11/site-packages/torch \
       -type d \( -name tests -o -name test \) \
       -prune -exec rm -rf {} +

COPY service/ /app/service/
COPY models/v2/ /app/models/v2/

EXPOSE 8000

CMD ["python", "service/app.py"]
