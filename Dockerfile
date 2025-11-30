# =====================================================
# Dockerfile para API Unificada ML - v2.0.0
# Funciona en LOCAL (desarrollo) y RAILWAY (producción)
# =====================================================

# =====================================================
# Stage 1: Builder (compilar dependencias)
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
# Stage 2: Runtime (imagen final)
# =====================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/mluser/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ENVIRONMENT=production

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 mluser

# Copiar dependencias desde builder
COPY --from=builder /root/.local /home/mluser/.local
RUN chown -R mluser:mluser /home/mluser/.local

# Copiar código
COPY --chown=mluser:mluser . .

# Crear directorios necesarios
RUN mkdir -p /app/trained_models /app/logs && \
    chmod -R 755 /app && \
    chmod -R 777 /app/trained_models /app/logs && \
    chown -R mluser:mluser /app /home/mluser

# Health check (verifica que el servidor esté respondiendo)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

USER mluser

EXPOSE 8080

# Ejecutar servidor unificado
# - Puerto: 8080 (Railway automáticamente)
# - Host: 0.0.0.0 (accesible desde fuera)
# - Reload: false (producción no recarga)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
