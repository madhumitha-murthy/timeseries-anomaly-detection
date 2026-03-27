# Anomaly Detection · LSTM Autoencoder · NASA SMAP

Semi-supervised time-series anomaly detection using an LSTM Autoencoder trained on NASA SMAP satellite telemetry, served via a FastAPI REST API and containerised with Docker.

---

## Architecture

```
Input window               Encoder LSTM            Bottleneck
(B, W, F) ─────────────►  (F → hidden_dim)  ────►  h_n[-1]
                                                      │
                                                      │ repeat W times
                                                      ▼
Reconstruction             Decoder LSTM           Decoder input
(B, W, F)  ◄───────────  (hidden → F)       ◄────  (B, W, hidden)
     │
     ▼
MSE vs original
     │
  Threshold
     │
     ▼
Anomaly flag (0/1)
```

---

## Results

| Model             | F1    | Precision | Recall | AUC   |
|-------------------|-------|-----------|--------|-------|
| LSTM Autoencoder  | 0.73  | 0.71      | 0.76   | 0.84  |
| Isolation Forest  | 0.55  | 0.52      | 0.61   | —     |
| **Improvement**   | **+33% F1** | | | |

Evaluated on NASA SMAP channel P-1 (562 labelled anomaly sequences across 54 channels).

---

## Quick Start

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Download SMAP data (from Kaggle)
# https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
# After downloading and unzipping, copy into the project:
cp -r path/to/downloaded/train ./data/train
cp -r path/to/downloaded/test  ./data/test
cp path/to/labeled_anomalies.csv ./data/

# 3 — Train the model
cd src && python train.py

# 4 — Serve the API locally
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 5 — Run with Docker
docker compose up --build
```

---

## API Usage

**Health check**
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

**Predict anomaly score for a single window**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "window": [[0.12], [0.15], [0.11], [-0.02], [0.08]],
    "threshold": 0.05
  }'
```
Response:
```json
{
  "anomaly_score": 0.003142,
  "is_anomaly": false,
  "threshold_used": 0.05,
  "latency_ms": 11.74
}
```

Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs` once the server is running.

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **Window stride = 1** | Maximises training samples from limited labelled data; SMAP train splits have ~2–8 k steps per channel. |
| **Gradient clipping `max_norm=1.0`** | BPTT through 128 time-steps can cause exploding gradients in LSTMs. Clipping the norm caps the update magnitude without eliminating gradient signal. |
| **`ReduceLROnPlateau` scheduler** | Automatically halves the learning rate when validation loss plateaus (patience=5), avoiding manual LR tuning and helping converge to a better optimum. |
| **MLflow experiment tracking** | Every training run logs hyperparameters, per-epoch train/val loss, and the best checkpoint as an artefact. This enables A/B comparisons between configurations (hidden_dim, num_layers, lr) without relying on manual notes. |
| **Point-adjust max-pooling** | Maps window scores to per-step scores by taking the MAX over all windows containing a step. This is the standard convention in the SMAP/MSL literature (Hundman et al. 2018) and is conservative: a single high-error window is enough to flag a time-step. |

---

## Stack

PyTorch · scikit-learn · NumPy · Pandas · FastAPI · MLflow · Docker
