"""
Configuración centralizada para API Unificada
Soporta local (development) y producción (Railway)
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ============================================================
# AMBIENTE
# ============================================================

ENVIRONMENT = os.getenv('ENVIRONMENT', 'development').lower()
IS_PRODUCTION = ENVIRONMENT in ('production', 'railway')
IS_DEVELOPMENT = not IS_PRODUCTION

DEBUG = os.getenv('DEBUG', 'true').lower() == 'true' if IS_DEVELOPMENT else False
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG' if IS_DEVELOPMENT else 'INFO').upper()

# ============================================================
# PUERTO
# ============================================================

if IS_PRODUCTION:
    # Railway asigna el puerto en la variable PORT
    PORT = int(os.getenv('PORT', 8080))
else:
    # Local usa 8001
    PORT = int(os.getenv('PORT', 8001))

HOST = os.getenv('HOST', '0.0.0.0')

# ============================================================
# BASE DE DATOS
# ============================================================

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_DATABASE', 'educativa')
DB_USER = os.getenv('DB_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '1234')

# ============================================================
# FEATURES Y AUTENTICACIÓN
# ============================================================

# En producción: activar autenticación Sanctum
# En desarrollo: desactivar para facilitar testing
ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'true' if IS_PRODUCTION else 'false').lower() == 'true'

# En producción: activar caché avanzado
# En desarrollo: desactivar para mayor velocidad
ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'true' if IS_PRODUCTION else 'false').lower() == 'true'

# En producción: activar agente IA para recomendaciones
# En desarrollo: desactivar si no está disponible
ENABLE_AGENT = os.getenv('ENABLE_AGENT', 'true' if IS_PRODUCTION else 'false').lower() == 'true'

# Endpoint vocacional: disponible en ambas versiones
ENABLE_VOCATIONAL = os.getenv('ENABLE_VOCATIONAL', 'true').lower() == 'true'

# Endpoint de dificultad de preguntas: disponible en ambas versiones
ENABLE_QUESTION_DIFFICULTY = os.getenv('ENABLE_QUESTION_DIFFICULTY', 'true').lower() == 'true'

# Habilitar predicción por lotes: disponible en ambas versiones
ENABLE_BATCH = os.getenv('ENABLE_BATCH', 'true').lower() == 'true'

# CORS: siempre habilitar en producción, opcional en desarrollo
ENABLE_CORS = os.getenv('ENABLE_CORS', 'true' if IS_PRODUCTION else 'false').lower() == 'true'

# ============================================================
# MODELOS
# ============================================================

from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'trained_models'

# ============================================================
# SANCTUM (AUTENTICACIÓN)
# ============================================================

LARAVEL_APP_KEY = os.getenv('LARAVEL_APP_KEY') or os.getenv('APP_KEY')

# ============================================================
# INFORMACIÓN DEL SERVIDOR
# ============================================================

API_TITLE = "Plataforma Educativa - ML Prediction API"
API_VERSION = "2.0.0"
API_DESCRIPTION = "Servidor unificado de predicciones ML para análisis educativo (Local + Producción)"

# ============================================================
# RESUMEN DE CONFIGURACIÓN
# ============================================================

CONFIG_SUMMARY = {
    'environment': ENVIRONMENT,
    'is_production': IS_PRODUCTION,
    'debug': DEBUG,
    'port': PORT,
    'host': HOST,
    'features': {
        'auth': ENABLE_AUTH,
        'cache': ENABLE_CACHE,
        'agent': ENABLE_AGENT,
        'vocational': ENABLE_VOCATIONAL,
        'question_difficulty': ENABLE_QUESTION_DIFFICULTY,
        'batch': ENABLE_BATCH,
        'cors': ENABLE_CORS,
    },
    'database': {
        'host': DB_HOST,
        'port': DB_PORT,
        'name': DB_NAME,
    }
}
