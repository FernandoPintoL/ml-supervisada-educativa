# Configuración Global
# Plataforma Educativa ML

import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================

PROJECT_NAME = "Plataforma Educativa ML"
PROJECT_VERSION = "2.0.0"
DEBUG = os.getenv("DEBUG", "False") == "True"

# ==========================================
# BASE DE DATOS
# ==========================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/educativa_db"
)
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = int(os.getenv("DATABASE_PORT", 5432))
DATABASE_NAME = os.getenv("DATABASE_NAME", "educativa_db")
DATABASE_USER = os.getenv("DATABASE_USER", "user")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "password")

# ==========================================
# REDIS / CACHÉ
# ==========================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# ==========================================
# API
# ==========================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "True") == "True"
API_WORKERS = int(os.getenv("API_WORKERS", 4))

# ==========================================
# ML - THRESHOLDS Y PARÁMETROS
# ==========================================

# Predictor de Desempeño
PERFORMANCE_RISK_THRESHOLD_HIGH = 0.70  # > 70% = Riesgo Alto
PERFORMANCE_RISK_THRESHOLD_MEDIUM = 0.40  # 40-70% = Riesgo Medio

# Detector de Plagio
PLAGIARISM_THRESHOLD_CRITICAL = 0.80  # > 80% = Definido plagio
PLAGIARISM_THRESHOLD_WARNING = 0.70  # 70-80% = Probable plagio
PLAGIARISM_THRESHOLD_CAUTION = 0.60  # 60-70% = Revisar

# Clustering
DEFAULT_N_CLUSTERS = 4  # K-Means clusters
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 10

# Anomalía Detection
ANOMALY_CONTAMINATION = 0.1  # 10% de datos son anomalías
ANOMALY_THRESHOLD = 0.5

# ==========================================
# LOGGING
# ==========================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/ml_system.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==========================================
# DIRECTORIOS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Asegurar que existan los directorios
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ==========================================
# DEEP LEARNING
# ==========================================

USE_GPU = os.getenv("USE_GPU", "True") == "True"
GPU_DEVICE = int(os.getenv("GPU_DEVICE", 0))
TF_RANDOM_SEED = int(os.getenv("TF_RANDOM_SEED", 42))

# ==========================================
# MODELOS SUPERVISADOS
# ==========================================

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 5
RF_RANDOM_STATE = 42

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.1
XGB_RANDOM_STATE = 42
XGB_SCALE_POS_WEIGHT = 1

# SVM
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_PROBABILITY = True

# KNN
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"

# ==========================================
# VALIDACIÓN
# ==========================================

TEST_SIZE = 0.2  # 80% train, 20% test
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# ==========================================
# DATOS
# ==========================================

MIN_STUDENTS_REQUIRED = 100  # Mínimo para entrenar supervisado
MIN_STUDENTS_UNSUPERVISED = 200  # Mínimo para no supervisado
MIN_STUDENTS_DEEP_LEARNING = 10000  # Mínimo para deep learning

# ==========================================
# SEGURIDAD
# ==========================================

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Qué features usar (editable según necesidad)
SUPERVISADO_FEATURES = [
    "promedio_ultimas_notas",
    "varianza_notas",
    "asistencia_porcentaje",
    "trabajos_entregados_tarde",
    "horas_estudio_semanal"
]

NO_SUPERVISADO_FEATURES = [
    "promedio",
    "asistencia",
    "velocidad_aprendizaje",
    "consistencia_notas",
    "participacion"
]

DEEP_LEARNING_FEATURES = [
    "secuencia_notas",
    "embeddings",
    "temporal_patterns"
]

# ==========================================
# PRINT CONFIGURATION (Para debug)
# ==========================================

if DEBUG:
    print(f"[Config] Database: {DATABASE_URL}")
    print(f"[Config] Redis: {REDIS_URL}")
    print(f"[Config] API: {API_HOST}:{API_PORT}")
    print(f"[Config] Models Dir: {MODELS_DIR}")
    print(f"[Config] Logs Dir: {LOGS_DIR}")
