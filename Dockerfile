FROM python:3.11-slim

# Make Python logs unbuffered and avoid pip cache
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Work inside /app
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL app code (includes app.py and any helpers)
COPY . .

# Gradio UI listens on 7860 (or $PORT), Prometheus on 8000
EXPOSE 7860
EXPOSE 8000

# Start the app
CMD ["python", "app.py"]
