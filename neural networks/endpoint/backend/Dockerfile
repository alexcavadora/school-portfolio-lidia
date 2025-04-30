FROM python:3.9-slim

WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app /app

ENV PYTHONPATH=/app

EXPOSE 3000

CMD ["uvicorn", "mnist_server:app", "--host", "0.0.0.0", "--port", "3000"]