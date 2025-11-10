FROM python:3.10-slim

# Install Node Exporter (system metrics for Prometheus)
RUN apt-get update && apt-get install -y --no-install-recommends \
    prometheus-node-exporter \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app
ENV NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    http_proxy= \
    https_proxy=


# Install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose:
#  - 7860 : Gradio UI
#  - 8000 : App metrics (prometheus_client)
#  - 9100 : Node Exporter metrics
EXPOSE 7860 8000 9100

# Start Node Exporter in background, then your app (professor style)
CMD prometheus-node-exporter --web.listen-address=":9100" & python3 app.py
