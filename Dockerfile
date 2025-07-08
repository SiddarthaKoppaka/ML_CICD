# Base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1
# Entrypoint for training
CMD ["python", "training_pipeline/train.py", "--config", "training_pipeline/config.yaml"]