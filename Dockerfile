FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and pre-trained model artefacts
COPY src/ ./src/
COPY models/ ./models/

# Runtime configuration — override at docker run / compose time as needed
ENV MODEL_PATH=/app/models/lstm_ae_best.pth \
    INPUT_DIM=25 \
    HIDDEN_DIM=64 \
    NUM_LAYERS=1 \
    THRESHOLD=0.05

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
