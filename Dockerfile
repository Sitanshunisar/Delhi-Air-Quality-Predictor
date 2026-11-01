# Dockerfile — Streamlit app with Prometheus metrics exporter
FROM python:3.11-slim

WORKDIR /app

# Install system deps for TF / LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["python", "run_app.py"]
