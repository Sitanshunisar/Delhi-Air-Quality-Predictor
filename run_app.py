# run_app.py
"""Entrypoint to guarantee Prometheus metrics exporter starts immediately on container boot."""
import sys

# 1. Start Prometheus HTTP Server on 0.0.0.0:8000
try:
    from prometheus_client import start_http_server

    start_http_server(8000, addr="0.0.0.0")
    print("Prometheus metrics exporter successfully started on 0.0.0.0:8000", flush=True)
except Exception as exc:
    print(f"Metrics exporter startup note: {exc}", flush=True)

# 2. Launch Streamlit app on 0.0.0.0:8501 in the same Python process
from streamlit.web import cli as stcli

sys.argv = [
    "streamlit",
    "run",
    "app/streamlit_app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0",
]

if __name__ == "__main__":
    sys.exit(stcli.main())

