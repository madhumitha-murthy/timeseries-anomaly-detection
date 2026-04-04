FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and pre-trained model artefacts
COPY src/ ./src/
COPY models/ ./models/

# Runtime configuration — override at docker run / compose time as needed.
# THRESHOLD is intentionally omitted: the API reads the value from
# threshold.json (saved by train.py) at startup.  Only set THRESHOLD here
# if you need to manually override the trained value.
ENV MODEL_PATH=/app/models/lstm_ae_best.pth \
    THRESHOLD_PATH=/app/models/threshold.json \
    INPUT_DIM=25 \
    HIDDEN_DIM=64 \
    NUM_LAYERS=1 \
    WINDOW_SIZE=30

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
