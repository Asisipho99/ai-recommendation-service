FROM python:3.11-slim

WORKDIR /app

# Install gcc and build tools needed for scikit-surprise C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create model directory
RUN mkdir -p model

# Railway injects $PORT at runtime — default to 5000 for local
ENV PORT=5000

EXPOSE ${PORT}

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 120 app:app"]
