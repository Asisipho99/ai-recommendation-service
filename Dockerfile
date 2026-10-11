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

# Bake the source dataset into the image.
#
# .gitignore excludes data/raw/*, so the 15MB cached CSV never reaches the
# build context and the deployed service previously had no dataset at all -
# /api/data/source reported available:false and the whole EDA and training
# pipeline was dead in production while working fine locally.
#
# Fetching at build rather than at runtime matters because the container
# filesystem is ephemeral: anything downloaded at runtime is lost on the next
# deploy, whereas an image layer is not.
#
# Deliberately non-fatal. UCI occasionally rate-limits, and an unrelated
# bug fix should still be deployable when it does. POST /api/data/download
# is the recovery path when this line fails.
RUN python scripts/download_dataset.py || echo "WARNING: build-time dataset download failed; call POST /api/data/download on the running service"

# Railway injects $PORT at runtime — default to 5000 for local
ENV PORT=5000

EXPOSE ${PORT}

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 120 app:app"]
