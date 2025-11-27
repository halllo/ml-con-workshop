FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code
COPY recommender.py .
COPY serve.py .

# Create model directory
RUN mkdir -p /mnt/models

# Expose port
EXPOSE 8080

# Run server
CMD ["python", "serve.py"]
