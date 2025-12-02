FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Gradio UI + Prometheus metrics (inside the container)
EXPOSE 7860 8000

# OCRSPACE_API_KEY will be injected by the cloud provider as an env var
CMD ["python", "app.py"]
