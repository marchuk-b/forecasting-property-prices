FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# xgboost needs the OpenMP runtime, which is not included in the slim image
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first: this layer is cached until requirements change
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Project files (see .dockerignore: trained models are NOT copied, they live in a volume)
COPY data ./data
COPY src ./src
COPY app ./app
COPY tests ./tests
COPY models ./models

# Run as a normal user, not root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app.main:app"]
