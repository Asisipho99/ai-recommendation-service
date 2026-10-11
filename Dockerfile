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

# --timeout 300, raised from 120 for headroom rather than necessity.
#
# Measured on the full 1,067,371-row dataset: load 3.5s, EDA 6.1s, prepare
# 10.7s, so the worst synchronous request is about 14s and 120 was never
# actually at risk. 300 is roughly twenty times the measured figure, which
# covers a slower shared vCPU and two workers competing, without letting a
# genuinely wedged worker sit unreaped for a quarter of an hour.
#
# The reason any of this needs stating: /api/data/prepare and /api/ml/train
# are synchronous, and gunicorn kills a worker that has not responded within
# --timeout, which surfaces as a dropped connection partway through rather
# than as a slow success.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 300 --graceful-timeout 60 app:app"]
