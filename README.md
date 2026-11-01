# 🌬️ Delhi AQI Predictor

> Predict Delhi PM2.5 air quality using an ensemble of ML models (StackingRegressor, LSTM, GRU), served via **Streamlit**, with full CI/CD, Docker, Kubernetes, and observability.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
python -m src.train
```

### 3. Run Streamlit app
```bash
streamlit run app/streamlit_app.py
```

## Docker

```bash
docker-compose up --build
```

| Service    | URL                   |
|------------|-----------------------|
| Streamlit  | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana    | http://localhost:3000 |

## CI/CD (GitHub Actions)

| Stage        | Trigger         | Action                      |
|--------------|-----------------|-----------------------------|
| Test         | Every push / PR | `pytest tests/`             |
| Lint         | Every push / PR | `ruff check`                |
| Build & Push | Merge → `main`  | Docker image → GHCR         |
| Deploy       | Merge → `main`  | `kubectl apply` to GKE      |

Set GitHub Secret `KUBE_CONFIG` (base64-encoded kubeconfig) to enable the deploy job.

## Kubernetes

```bash
kubectl apply -f infra/k8s/
kubectl get svc delhi-aqi-service   # Get external IP
```

## Terraform (GCP / GKE)

```bash
cd infra/terraform
terraform init
terraform plan  -var="project_id=YOUR_GCP_PROJECT"
terraform apply -var="project_id=YOUR_GCP_PROJECT"
```

## Monitoring

Grafana runs at http://localhost:3000 (admin / admin).
Import `monitoring/grafana-dashboard.json` to load the preconfigured dashboard.

## Models

| Model             | Type        | R²     | Use case             |
|-------------------|-------------|--------|----------------------|
| StackingRegressor | Inference   | 0.9618 | Real-time inference  |
| GRU               | Forecasting | 0.8905 | Sequence forecasting |
| LSTM              | Forecasting | 0.8359 | Sequence forecasting |

## Project Structure

```
delhi-aqi-predictor/
├── data/                   Raw CSV data
├── models/                 Trained model artifacts
├── src/                    Core ML library
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── streamlit_app.py    Streamlit UI
├── notebooks/              Exploratory analysis
├── tests/                  Pytest test suite
├── infra/
│   ├── terraform/          GCP infrastructure
│   └── k8s/                Kubernetes manifests
├── monitoring/             Prometheus + Grafana
├── .github/workflows/      GitHub Actions CI/CD
├── Dockerfile              Streamlit container
└── docker-compose.yml      Local full stack
```
