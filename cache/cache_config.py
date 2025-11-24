"""
Configuración para el sistema de caché de datasets.
"""

import os
from pathlib import Path

# Directorio base del caché
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "trained_models" / "cache"

# Configuración de expiración del caché
CACHE_MAX_AGE_HOURS = int(os.getenv("CACHE_MAX_AGE_HOURS", "24"))

# Habilitar/deshabilitar caché
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# Usar caché si está disponible (pero no requerirlo)
CACHE_OPTIONAL = os.getenv("CACHE_OPTIONAL", "false").lower() == "true"

# Estrategias de caché
CACHE_STRATEGY = os.getenv("CACHE_STRATEGY", "lazy")  # "lazy" o "eager"
# - lazy: Carga desde caché solo si existe, si no carga de BD
# - eager: Intenta caché primero, falla si no existe

# Logging de acceso a caché
CACHE_LOG_HITS = os.getenv("CACHE_LOG_HITS", "true").lower() == "true"

# Configuración de compresión
COMPRESS_CACHE = os.getenv("COMPRESS_CACHE", "false").lower() == "true"
