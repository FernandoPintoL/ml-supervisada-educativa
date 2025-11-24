# =====================================================
# Stage 1: Builder
# =====================================================

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt

# =====================================================
# Stage 2: Runtime
# =====================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/mluser/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 mluser

COPY --from=builder /root/.local /home/mluser/.local
RUN chown -R mluser:mluser /home/mluser/.local

COPY --chown=mluser:mluser . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

USER mluser

EXPOSE 8080

# Entry point: uvicorn en puerto 8080 para Railway
CMD sh -c 'uvicorn api_server:app --host 0.0.0.0 --port 8080'
