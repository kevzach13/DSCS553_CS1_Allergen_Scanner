FROM python:3.11-slim

# Work inside /app
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# (Optional) just so logs flush immediately
ENV PYTHONUNBUFFERED=1

# Cloud Run sends traffic to $PORT (usually 8080)
# EXPOSE is just documentation, but we set it to 8080 to avoid confusion.
EXPOSE 8080

# OCRSPACE_API_KEY and PORT will be injected by Cloud Run as env vars
CMD ["python", "app.py"]
