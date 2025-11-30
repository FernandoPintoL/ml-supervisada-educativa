"""
FastAPI Prediction Server Unificado para Plataforma Educativa
Versión 2.0.0 - Soporta local (development) y producción (Railway)

Uso:
    Local:      python api_server.py
    Producción: gunicorn api_server:app

Configuración por variables de entorno (ver config.py)
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Agregar directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Importar configuración centralizada
from config import (
    ENVIRONMENT, IS_PRODUCTION, IS_DEVELOPMENT, DEBUG, LOG_LEVEL,
    PORT, HOST, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    ENABLE_AUTH, ENABLE_CACHE, ENABLE_AGENT, ENABLE_VOCATIONAL,
    ENABLE_QUESTION_DIFFICULTY, ENABLE_BATCH, ENABLE_CORS,
    MODELS_DIR, LARAVEL_APP_KEY, API_TITLE, API_VERSION, API_DESCRIPTION,
    CONFIG_SUMMARY
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# IMPORTACIONES CONDICIONALES
# ============================================================

# Autenticación Sanctum (solo en producción)
if ENABLE_AUTH:
    try:
        from auth.sanctum_auth import (
            init_sanctum_auth,
            get_authenticator,
            SanctumRoleValidator,
            token_cache,
        )
        logger.info("✓ Módulo Sanctum Auth importado")
    except ImportError:
        logger.warning("⚠ Módulo Sanctum Auth no disponible")
        ENABLE_AUTH = False
else:
    # Mock para desarrollo
    token_cache = None
    SanctumRoleValidator = None

# Caché avanzado (solo en producción)
if ENABLE_CACHE:
    try:
        from cache.cache_manager import get_cache_manager, load_cached_dataset
        logger.info("✓ Módulo Cache Manager importado")
    except ImportError:
        logger.warning("⚠ Módulo Cache Manager no disponible")
        ENABLE_CACHE = False

# DataProcessor (para procesamiento avanzado)
try:
    from data.data_processor import DataProcessor
    logger.info("✓ Módulo DataProcessor importado")
except ImportError:
    logger.warning("⚠ Módulo DataProcessor no disponible")
    DataProcessor = None

# Modelos ML
try:
    from models.performance_predictor import PerformancePredictor
    from models.career_recommender import CareerRecommender
    from models.trend_predictor import TrendPredictor, TREND_LABELS
    from models.progress_analyzer import ProgressAnalyzer
    logger.info("✓ Módulos ML importados")
except ImportError as e:
    logger.warning(f"⚠ Error importando modelos ML: {e}")

# Agente IA para recomendaciones (solo en producción)
if ENABLE_AGENT:
    try:
        from agents.recommendation_agent import get_agent
        logger.info("✓ Módulo Recommendation Agent importado")
    except ImportError:
        logger.warning("⚠ Módulo Recommendation Agent no disponible")
        ENABLE_AGENT = False

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class StudentData(BaseModel):
    """Datos del estudiante para predicción"""
    student_id: int
    promedio_calificaciones: float
    varianza_calificaciones: float
    max_calificacion: float
    min_calificacion: float
    num_calificaciones: int
    num_trabajos: int
    promedio_intentos: float
    dias_promedio_entrega: float
    promedio_consultas_material: float
    trabajos_entregados: int
    trabajos_calificados: int


class RiskPredictionResponse(BaseModel):
    """Respuesta de predicción de riesgo"""
    student_id: int
    risk_level: str
    risk_score: float
    confidence: float
    timestamp: str


class CareerPredictionResponse(BaseModel):
    """Respuesta de recomendación de carrera"""
    student_id: int
    top_3_careers: List[Dict[str, Any]]
    timestamp: str


class TrendPredictionResponse(BaseModel):
    """Respuesta de predicción de tendencia"""
    student_id: int
    trend: str
    confidence: float
    timestamp: str


class ProgressPredictionResponse(BaseModel):
    """Respuesta de predicción de progreso"""
    student_id: int
    projected_grade: float
    learning_velocity: float
    acceleration: float
    confidence: float
    timestamp: str


class VocationalFeaturesRequest(BaseModel):
    """Features para predicción vocacional"""
    student_id: int
    promedio: float
    asistencia: float
    tasa_entrega: float
    tendencia_score: float
    recencia_score: float
    area_dominante: float
    num_areas_fuertes: int


class VocationalCareerResponse(BaseModel):
    """Respuesta de predicción vocacional de carrera"""
    student_id: int
    carrera: str
    confianza: float
    compatibilidad: float
    top_3: List[Dict[str, Any]]
    modelo_version: str
    tiempo_procesamiento_ms: float
    timestamp: str


class QuestionDifficultyRequest(BaseModel):
    """Solicitud para predecir dificultad de pregunta"""
    enunciado: str
    tipo: str
    nivel_bloom: str
    curso_id: Optional[int] = None
    longitud_opciones: Optional[int] = None


class QuestionDifficultyResponse(BaseModel):
    """Respuesta con predicción de dificultad"""
    dificultad_predicha: float
    confianza: float
    clasificacion: str
    razonamiento: str


class HealthCheckResponse(BaseModel):
    """Respuesta de verificación de salud"""
    status: str
    environment: str
    models_loaded: Dict[str, bool]
    features_enabled: Dict[str, bool]
    timestamp: str


class RecommendationRequest(BaseModel):
    """Request para generar recomendaciones"""
    student_id: int
    name: str
    subject: str
    current_grade: float
    previous_average: float = 0.0
    num_calificaciones: int
    num_trabajos: int
    trabajos_entregados: int
    dias_promedio_entrega: float
    promedio_consultas_material: float
    risk_score: float
    risk_level: str
    projected_grade: float
    trend: str
    confidence: float


class RecommendationResponse(BaseModel):
    """Respuesta con recomendaciones educativas"""
    status: str
    student_id: int
    recommendation_type: str
    urgency: str
    reason: str
    actions: List[str]
    resources: List[str]
    success_indicators: List[str]
    confidence_level: str
    timestamp: str


class PredictionRequest(BaseModel):
    """Solicitud de predicción simple"""
    student_id: int


class PredictionResponse(BaseModel):
    """Respuesta de predicción simple"""
    student_id: int
    prediction: float
    confidence: Optional[float] = None
    model_name: str


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Configuration (condicional según ambiente)
if ENABLE_CORS or IS_PRODUCTION:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ============================================================
# DATABASE CONNECTION
# ============================================================

class DBConnection:
    """Conexión a Base de Datos PostgreSQL"""

    @staticmethod
    def connect():
        """Crear conexión a BD"""
        try:
            return psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
        except Exception as e:
            logger.error(f"[ERROR] Conexión BD fallida: {e}")
            return None

    @staticmethod
    def get_student_data(student_id: int) -> Optional[Dict]:
        """Obtener datos de un estudiante"""
        conn = DBConnection.connect()
        if not conn:
            return None

        query = """
        SELECT
            u.id as estudiante_id,
            u.desempeño_promedio,
            u.asistencia_porcentaje,
            u.participacion_porcentaje,
            u.tareas_completadas,
            u.tareas_pendientes,
            u.actividad_hoy,
            COALESCE(AVG(c.puntaje), 0) as promedio_calificaciones,
            COUNT(c.id) as total_calificaciones,
            COALESCE(ra.promedio, 0) as promedio_rendimiento
        FROM users u
        LEFT JOIN trabajos t ON u.id = t.estudiante_id
        LEFT JOIN calificaciones c ON t.id = c.trabajo_id
        LEFT JOIN rendimiento_academico ra ON u.id = ra.estudiante_id
        WHERE u.id = %s AND u.tipo_usuario = 'estudiante'
        GROUP BY u.id, u.desempeño_promedio, u.asistencia_porcentaje,
                 u.participacion_porcentaje, u.tareas_completadas,
                 u.tareas_pendientes, u.actividad_hoy, ra.promedio
        """

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, (student_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[ERROR] Query BD fallida: {e}")
            return None


# ============================================================
# MODEL MANAGER
# ============================================================

class ModelManager:
    """Gestiona carga y predicciones de modelos ML"""

    def __init__(self):
        self.models = {}
        self.processor = DataProcessor(scaler_type="standard") if DataProcessor else None
        self.is_ready = False
        self.cache_manager = get_cache_manager() if ENABLE_CACHE else None
        self.cached_data = None
        self.cached_features = None
        self.cache_loaded = False

    def load_models(self):
        """Cargar todos los modelos entrenados"""
        logger.info("\n" + "=" * 60)
        logger.info("CARGANDO MODELOS ML")
        logger.info("=" * 60)

        try:
            # Cargar Performance Predictor (Risk)
            logger.info("  Cargando PerformancePredictor...")
            risk_model_path = MODELS_DIR / 'PerformancePredictor_model.pkl'
            if risk_model_path.exists():
                self.models['risk'] = PerformancePredictor()
                self.models['risk'].load(str(risk_model_path))
                logger.info("    ✓ PerformancePredictor cargado")
            else:
                logger.warning(f"    ✗ No encontrado: {risk_model_path}")

            # Cargar Career Recommender
            logger.info("  Cargando CareerRecommender...")
            career_model_path = MODELS_DIR / 'CareerRecommender_model.pkl'
            if career_model_path.exists():
                self.models['career'] = CareerRecommender()
                self.models['career'].load(str(career_model_path))
                logger.info("    ✓ CareerRecommender cargado")
            else:
                logger.warning(f"    ✗ No encontrado: {career_model_path}")

            # Cargar Trend Predictor
            logger.info("  Cargando TrendPredictor...")
            trend_model_path = MODELS_DIR / 'TrendPredictor_model.pkl'
            if trend_model_path.exists():
                self.models['trend'] = TrendPredictor()
                self.models['trend'].load(str(trend_model_path))
                logger.info("    ✓ TrendPredictor cargado")
            else:
                logger.warning(f"    ✗ No encontrado: {trend_model_path}")

            # Cargar Progress Analyzer
            logger.info("  Cargando ProgressAnalyzer...")
            progress_model_path = MODELS_DIR / 'ProgressAnalyzer_model.pkl'
            if progress_model_path.exists():
                self.models['progress'] = ProgressAnalyzer()
                self.models['progress'].load(str(progress_model_path))
                logger.info("    ✓ ProgressAnalyzer cargado")
            else:
                logger.warning(f"    ✗ No encontrado: {progress_model_path}")

            self.is_ready = len(self.models) > 0
            logger.info(f"✓ Cargados {len(self.models)}/4 modelos")

            # Intentar cargar caché si está habilitado
            if ENABLE_CACHE:
                self.load_cached_dataset()

            logger.info("=" * 60 + "\n")
            return self.is_ready

        except Exception as e:
            logger.error(f"✗ Error cargando modelos: {str(e)}", exc_info=True)
            return False

    def load_cached_dataset(self):
        """Cargar dataset de caché para acelerar predicciones"""
        if not ENABLE_CACHE or not self.cache_manager:
            return False

        try:
            logger.info("\n[Cache] Intentando cargar dataset desde caché...")
            cache_info = self.cache_manager.get_cache_info()

            if not cache_info.get('exists', False):
                logger.info("  ⚠ No hay caché disponible")
                return False

            result = self.cache_manager.load_dataset()
            if result:
                self.cached_data, self.cached_features = result
                self.cache_loaded = True
                logger.info(f"  ✓ Dataset cargado desde caché")
                return True
            else:
                logger.warning("  ✗ Error al cargar dataset desde caché")
                return False

        except Exception as e:
            logger.warning(f"  ⚠ Error cargando caché: {str(e)}")
            return False

    def prepare_features(self, student_data: StudentData) -> np.ndarray:
        """Preparar características del estudiante"""
        features = np.array([[
            student_data.promedio_calificaciones,
            student_data.varianza_calificaciones,
            student_data.max_calificacion,
            student_data.min_calificacion,
            student_data.num_calificaciones,
            student_data.num_trabajos,
            student_data.promedio_intentos,
            student_data.dias_promedio_entrega,
            student_data.promedio_consultas_material,
            student_data.trabajos_entregados,
            student_data.trabajos_calificados,
            0,  # role encoded (estudiante)
        ]], dtype=float)
        return features


# Inicializar manager global
model_manager = ModelManager()


# ============================================================
# AUTENTICACIÓN (CONDICIONAL)
# ============================================================

async def verify_sanctum_token(authorization: str = Header(None)) -> Dict:
    """Verificar token de Sanctum (solo en producción)"""

    # En desarrollo, permitir sin token
    if not ENABLE_AUTH:
        return {"token_id": "dev_token", "user_id": 0, "valid": True}

    if not authorization:
        logger.warning("Request sin token Authorization")
        raise HTTPException(status_code=401, detail="Token requerido")

    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Formato de token inválido")

        token = authorization[7:]

        # Verificar en cache primero
        if token_cache:
            cached = token_cache.get(token)
            if cached:
                return cached

        # Validar token con Sanctum
        authenticator = get_authenticator()
        token_data = authenticator.validate_sanctum_token(token)

        if not token_data or not token_data.get("valid"):
            logger.warning("Token inválido")
            raise HTTPException(status_code=401, detail="Token inválido")

        # Guardar en cache
        if token_cache:
            token_cache.set(token, token_data)

        return token_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validando token: {str(e)}")
        raise HTTPException(status_code=401, detail="Error validando token")


# ============================================================
# STARTUP & SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio"""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 INICIANDO SERVIDOR API ML UNIFICADO")
    logger.info("=" * 70)
    logger.info(f"Ambiente: {ENVIRONMENT.upper()}")
    logger.info(f"Puerto: {PORT}")
    logger.info(f"Versión: {API_VERSION}")
    logger.info("-" * 70)

    # Mostrar features habilitadas
    logger.info("Features habilitados:")
    for feature, enabled in CONFIG_SUMMARY['features'].items():
        status = "✓" if enabled else "✗"
        logger.info(f"  {status} {feature}")

    logger.info("-" * 70)

    # Inicializar autenticación Sanctum si está habilitada
    if ENABLE_AUTH and LARAVEL_APP_KEY:
        try:
            init_sanctum_auth(LARAVEL_APP_KEY)
            logger.info("✓ Autenticación Sanctum inicializada")
        except Exception as e:
            logger.error(f"✗ Error inicializando Sanctum: {str(e)}")

    # Cargar modelos
    model_manager.load_models()
    if model_manager.is_ready:
        logger.info("✓ Servidor listo para recibir predicciones")
    else:
        logger.warning("⚠ Servidor iniciado pero sin modelos cargados")

    logger.info("=" * 70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre"""
    logger.info("✓ Servidor API ML cerrado")


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Verificar salud del servidor y carga de modelos"""
    models_loaded = {
        'risk': 'risk' in model_manager.models,
        'career': 'career' in model_manager.models,
        'trend': 'trend' in model_manager.models,
        'progress': 'progress' in model_manager.models,
    }

    status = "healthy" if model_manager.is_ready else "unhealthy"

    return HealthCheckResponse(
        status=status,
        environment=ENVIRONMENT,
        models_loaded=models_loaded,
        features_enabled=CONFIG_SUMMARY['features'],
        timestamp=datetime.now().isoformat()
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    """Endpoint raíz con información del servidor"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "status": "healthy" if model_manager.is_ready else "models_loading",
        "models_loaded": len(model_manager.models),
        "features_enabled": CONFIG_SUMMARY['features'],
        "endpoints": {
            "health": "/health",
            "predict_risk": "/predict/risk",
            "predict_career": "/predict/career",
            "predict_vocational": "/predict/career/vocational" if ENABLE_VOCATIONAL else None,
            "predict_trend": "/predict/trend",
            "predict_progress": "/predict/progress",
            "predict_batch": "/predict/batch" if ENABLE_BATCH else None,
            "predict_question_difficulty": "/predict/question-difficulty" if ENABLE_QUESTION_DIFFICULTY else None,
            "recommendations": "/recommendations" if ENABLE_AGENT else None,
            "cache_info": "/cache/info" if ENABLE_CACHE else None,
            "cache_refresh": "/cache/refresh" if ENABLE_CACHE else None,
            "cache_clear": "/cache/clear" if ENABLE_CACHE else None,
            "docs": "/docs",
            "redoc": "/redoc",
        }
    }


# ============================================================
# PREDICTION ENDPOINTS - RISK
# ============================================================

@app.post("/predict/risk", response_model=RiskPredictionResponse)
async def predict_risk(
    student_data: StudentData,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """Predecir nivel de riesgo del estudiante"""
    logger.info(f"Predicción de riesgo solicitada (token: {token_data.get('token_id')})")

    if 'risk' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Risk model not loaded")

    try:
        X = model_manager.prepare_features(student_data)
        risk_score = model_manager.models['risk'].predict(X)[0]

        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        confidence = abs(risk_score - 0.5) * 2
        confidence = min(max(confidence, 0), 1)

        return RiskPredictionResponse(
            student_id=student_data.student_id,
            risk_level=risk_level,
            risk_score=float(risk_score),
            confidence=float(confidence),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en predict_risk: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# PREDICTION ENDPOINTS - CAREER
# ============================================================

@app.post("/predict/career", response_model=CareerPredictionResponse)
async def predict_career(
    student_data: StudentData,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """Recomendar carreras basado en desempeño académico"""
    logger.info(f"Predicción de carrera solicitada (token: {token_data.get('token_id')})")

    if 'career' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Career model not loaded")

    try:
        X = model_manager.prepare_features(student_data)
        predictions = model_manager.models['career'].predict(X)[0]
        probabilities = model_manager.models['career'].predict_proba(X)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]

        career_labels = {
            0: 'Ingeniería en Sistemas',
            1: 'Administración de Empresas',
            2: 'Psicología',
            3: 'Educación',
            4: 'Medicina',
            5: 'Derecho',
            6: 'Contabilidad',
            7: 'Ingeniería Civil',
        }

        top_3_careers = [
            {
                "rank": rank + 1,
                "career": career_labels.get(int(idx), f"Career {idx}"),
                "compatibility": float(probabilities[idx]),
            }
            for rank, idx in enumerate(top_3_indices)
        ]

        return CareerPredictionResponse(
            student_id=student_data.student_id,
            top_3_careers=top_3_careers,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en predict_career: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# PREDICTION ENDPOINTS - VOCATIONAL (CONDICIONAL)
# ============================================================

@app.post("/predict/career/vocational", response_model=VocationalCareerResponse)
async def predict_career_vocational(
    features: VocationalFeaturesRequest,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """Predicción de carrera vocacional (solo si está habilitado)"""
    if not ENABLE_VOCATIONAL:
        raise HTTPException(status_code=404, detail="Vocational endpoint not available")

    import time
    start_time = time.time()

    logger.info(f"Predicción vocacional solicitada (token: {token_data.get('token_id')})")

    if 'career' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Career model not loaded")

    try:
        academic_features = StudentData(
            student_id=features.student_id,
            promedio_calificaciones=features.promedio / 10.0,
            varianza_calificaciones=features.area_dominante / 10.0,
            max_calificacion=10.0,
            min_calificacion=features.promedio / 10.0 * 0.7,
            num_calificaciones=int(features.num_areas_fuertes * 3),
            num_trabajos=int(features.tasa_entrega * 10),
            promedio_intentos=features.recencia_score * 3,
            dias_promedio_entrega=30 * (1 - features.tasa_entrega),
            promedio_consultas_material=features.asistencia / 20,
            trabajos_entregados=int(features.tasa_entrega * 10),
            trabajos_calificados=int(features.tasa_entrega * 10),
        )

        X = model_manager.prepare_features(academic_features)
        predictions = model_manager.models['career'].predict(X)[0]
        probabilities = model_manager.models['career'].predict_proba(X)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]

        career_labels = {
            0: 'Ingeniería en Sistemas',
            1: 'Administración de Empresas',
            2: 'Psicología',
            3: 'Educación',
            4: 'Medicina',
            5: 'Derecho',
            6: 'Contabilidad',
            7: 'Ingeniería Civil',
        }

        top_3 = [
            {
                "ranking": rank + 1,
                "carrera": career_labels.get(int(idx), f"Career {idx}"),
                "confianza": float(probabilities[idx]),
                "compatibilidad": float(probabilities[idx] * (features.area_dominante / 100.0))
            }
            for rank, idx in enumerate(top_3_indices)
        ]

        main_career = top_3[0] if top_3 else {
            "ranking": 1,
            "carrera": "No determinada",
            "confianza": 0.0,
            "compatibilidad": 0.0
        }

        processing_time = (time.time() - start_time) * 1000

        return VocationalCareerResponse(
            student_id=features.student_id,
            carrera=main_career["carrera"],
            confianza=main_career["confianza"],
            compatibilidad=main_career["compatibilidad"],
            top_3=top_3,
            modelo_version=API_VERSION,
            tiempo_procesamiento_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en predict_career_vocational: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# PREDICTION ENDPOINTS - TREND
# ============================================================

@app.post("/predict/trend", response_model=TrendPredictionResponse)
async def predict_trend(
    student_data: StudentData,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """Predecir tendencia de aprendizaje"""
    logger.info(f"Predicción de tendencia solicitada (token: {token_data.get('token_id')})")

    if 'trend' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Trend model not loaded")

    try:
        X = model_manager.prepare_features(student_data)
        trend_idx = model_manager.models['trend'].predict(X)[0]
        probabilities = model_manager.models['trend'].predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        trend_map = {0: "improving", 1: "stable", 2: "declining", 3: "fluctuating"}
        trend = trend_map.get(int(trend_idx), "stable")

        return TrendPredictionResponse(
            student_id=student_data.student_id,
            trend=trend,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en predict_trend: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# PREDICTION ENDPOINTS - PROGRESS
# ============================================================

@app.post("/predict/progress", response_model=ProgressPredictionResponse)
async def predict_progress(
    student_data: StudentData,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """Predecir progreso académico"""
    logger.info(f"Predicción de progreso solicitada (token: {token_data.get('token_id')})")

    if 'progress' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Progress model not loaded")

    try:
        X = model_manager.prepare_features(student_data)
        projected_grade = model_manager.models['progress'].predict(X)[0]

        current_grade = student_data.promedio_calificaciones
        learning_velocity = (projected_grade - current_grade) / max(student_data.num_calificaciones, 1)
        acceleration = learning_velocity * (student_data.varianza_calificaciones / 10)
        confidence = min(student_data.num_calificaciones / 20, 1.0)

        return ProgressPredictionResponse(
            student_id=student_data.student_id,
            projected_grade=float(np.clip(projected_grade, 0, 10)),
            learning_velocity=float(learning_velocity),
            acceleration=float(acceleration),
            confidence=float(confidence),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en predict_progress: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# BATCH PREDICTION (CONDICIONAL)
# ============================================================

@app.post("/predict/batch")
async def predict_batch(
    students: List[StudentData],
    token_data: Dict = Depends(verify_sanctum_token)
):
    """Predecir para múltiples estudiantes"""
    if not ENABLE_BATCH:
        raise HTTPException(status_code=404, detail="Batch endpoint not available")

    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")

    try:
        results = {
            'predictions': [],
            'timestamp': datetime.now().isoformat()
        }

        for student_data in students:
            student_result = {
                'student_id': student_data.student_id,
                'risk': None,
                'career': None,
                'trend': None,
                'progress': None,
            }

            # Risk prediction
            if 'risk' in model_manager.models:
                try:
                    X = model_manager.prepare_features(student_data)
                    risk_score = model_manager.models['risk'].predict(X)[0]
                    if risk_score >= 0.7:
                        risk_level = "HIGH"
                    elif risk_score >= 0.4:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "LOW"
                    student_result['risk'] = {'level': risk_level, 'score': float(risk_score)}
                except Exception as e:
                    logger.warning(f"Risk prediction failed for student {student_data.student_id}: {e}")

            # Career prediction
            if 'career' in model_manager.models:
                try:
                    X = model_manager.prepare_features(student_data)
                    probabilities = model_manager.models['career'].predict_proba(X)[0]
                    top_idx = np.argmax(probabilities)
                    career_labels = {0: 'Ingeniería en Sistemas', 1: 'Administración', 2: 'Psicología',
                                   3: 'Educación', 4: 'Medicina', 5: 'Derecho', 6: 'Contabilidad', 7: 'Ingeniería Civil'}
                    student_result['career'] = {
                        'top': career_labels.get(int(top_idx), 'Unknown'),
                        'compatibility': float(probabilities[top_idx])
                    }
                except Exception as e:
                    logger.warning(f"Career prediction failed for student {student_data.student_id}: {e}")

            # Trend prediction
            if 'trend' in model_manager.models:
                try:
                    X = model_manager.prepare_features(student_data)
                    trend_idx = model_manager.models['trend'].predict(X)[0]
                    trend_map = {0: "improving", 1: "stable", 2: "declining", 3: "fluctuating"}
                    student_result['trend'] = trend_map.get(int(trend_idx), "stable")
                except Exception as e:
                    logger.warning(f"Trend prediction failed for student {student_data.student_id}: {e}")

            # Progress prediction
            if 'progress' in model_manager.models:
                try:
                    X = model_manager.prepare_features(student_data)
                    projected_grade = model_manager.models['progress'].predict(X)[0]
                    student_result['progress'] = {
                        'projected_grade': float(np.clip(projected_grade, 0, 10))
                    }
                except Exception as e:
                    logger.warning(f"Progress prediction failed for student {student_data.student_id}: {e}")

            results['predictions'].append(student_result)

        return results

    except Exception as e:
        logger.error(f"Error en predict_batch: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# QUESTION DIFFICULTY ENDPOINT (CONDICIONAL)
# ============================================================

@app.post("/predict/question-difficulty", response_model=QuestionDifficultyResponse)
async def predict_question_difficulty(request: QuestionDifficultyRequest):
    """Predecir la dificultad de una pregunta educativa"""
    if not ENABLE_QUESTION_DIFFICULTY:
        raise HTTPException(status_code=404, detail="Question difficulty endpoint not available")

    try:
        longitud_enunciado = len(request.enunciado.strip())
        num_palabras = len(request.enunciado.split())

        bloom_mapping = {
            'remember': 0,
            'understand': 1,
            'apply': 2,
            'analyze': 3,
            'evaluate': 4,
            'create': 5
        }
        bloom_score = bloom_mapping.get(request.nivel_bloom.lower(), 2)

        tipo_mapping = {
            'verdadero_falso': 0,
            'opcion_multiple': 1,
            'respuesta_corta': 2,
            'respuesta_larga': 3
        }
        tipo_score = tipo_mapping.get(request.tipo.lower(), 1)

        bloom_difficulty = bloom_score / 5.0

        if longitud_enunciado < 20 or longitud_enunciado > 300:
            longitud_difficulty = 0.8
        elif longitud_enunciado < 50 or longitud_enunciado > 200:
            longitud_difficulty = 0.6
        else:
            longitud_difficulty = 0.4

        tipo_difficulty = tipo_score / 3.0

        dificultad_predicha = (
            bloom_difficulty * 0.40 +
            longitud_difficulty * 0.30 +
            tipo_difficulty * 0.30
        )

        dificultad_predicha = max(0.0, min(1.0, dificultad_predicha))

        if dificultad_predicha < 0.2:
            clasificacion = "muy_facil"
        elif dificultad_predicha < 0.4:
            clasificacion = "facil"
        elif dificultad_predicha < 0.6:
            clasificacion = "media"
        elif dificultad_predicha < 0.8:
            clasificacion = "dificil"
        else:
            clasificacion = "muy_dificil"

        razones = []
        if bloom_score >= 4:
            razones.append(f"Nivel Bloom alto ({request.nivel_bloom})")
        if longitud_enunciado > 200:
            razones.append("Enunciado extenso")
        if tipo_score > 1:
            razones.append(f"Tipo de pregunta requiere más análisis ({request.tipo})")

        razonamiento = "; ".join(razones) if razones else "Dificultad promedio"
        confianza = 0.75 + (0.25 * abs(0.5 - dificultad_predicha))

        logger.info(
            f"[PREDICT] Dificultad: {dificultad_predicha:.3f}, "
            f"Clasificación: {clasificacion}"
        )

        return QuestionDifficultyResponse(
            dificultad_predicha=round(dificultad_predicha, 3),
            confianza=round(confianza, 3),
            clasificacion=clasificacion,
            razonamiento=razonamiento
        )

    except Exception as e:
        logger.error(f"[ERROR] Predicción de dificultad fallida: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RECOMMENDATIONS ENDPOINT (CONDICIONAL)
# ============================================================

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """Generar recomendaciones educativas personalizadas"""
    if not ENABLE_AGENT:
        raise HTTPException(status_code=404, detail="Recommendations endpoint not available")

    try:
        logger.info(f"Generating recommendations for student {request.student_id}")

        agent = get_agent()
        student_data = {
            'student_id': request.student_id,
            'name': request.name,
            'subject': request.subject,
            'current_grade': request.current_grade,
            'previous_average': request.previous_average,
            'num_calificaciones': request.num_calificaciones,
            'num_trabajos': request.num_trabajos,
            'trabajos_entregados': request.trabajos_entregados,
            'dias_promedio_entrega': request.dias_promedio_entrega,
            'promedio_consultas_material': request.promedio_consultas_material,
        }

        predictions = {
            'risk_score': request.risk_score,
            'risk_level': request.risk_level,
            'projected_grade': request.projected_grade,
            'trend': request.trend,
            'confidence': request.confidence,
        }

        agent_response = agent.generate_recommendations(student_data, predictions)

        response = RecommendationResponse(
            status="success",
            student_id=request.student_id,
            recommendation_type=agent_response.get('recommendation_type', 'tutoring'),
            urgency=agent_response.get('urgency', 'normal'),
            reason=agent_response.get('reason_detailed', agent_response.get('reason', '')),
            actions=agent_response.get('actions', []),
            resources=agent_response.get('resources', []),
            success_indicators=agent_response.get('success_indicators', []),
            confidence_level=agent_response.get('confidence_level', 'medio'),
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Recommendations generated for student {request.student_id}")
        return response

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CACHE ENDPOINTS (CONDICIONAL)
# ============================================================

@app.post("/cache/refresh")
async def refresh_cache(token_data: Dict = Depends(verify_sanctum_token)):
    """Refrescar el caché"""
    if not ENABLE_CACHE:
        raise HTTPException(status_code=404, detail="Cache endpoint not available")

    try:
        logger.info("Refreshing cache...")
        success = model_manager.load_cached_dataset()

        if success:
            cache_info = model_manager.cache_manager.get_cache_info()
            return {
                "status": "success",
                "message": "Cache refreshed successfully",
                "cache_info": cache_info,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "warning",
                "message": "Cache is not available or failed to load",
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error refreshing cache: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/info")
async def get_cache_info(token_data: Dict = Depends(verify_sanctum_token)):
    """Obtener información del caché"""
    if not ENABLE_CACHE:
        raise HTTPException(status_code=404, detail="Cache endpoint not available")

    try:
        cache_info = model_manager.cache_manager.get_cache_info()
        return {
            "status": "success",
            "cache": cache_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/clear")
async def clear_cache(token_data: Dict = Depends(verify_sanctum_token)):
    """Limpiar el caché"""
    if not ENABLE_CACHE:
        raise HTTPException(status_code=404, detail="Cache endpoint not available")

    try:
        logger.info("Clearing cache...")
        success = model_manager.cache_manager.clear_cache()
        model_manager.cache_loaded = False
        model_manager.cached_data = None
        model_manager.cached_features = None

        if success:
            return {
                "status": "success",
                "message": "Cache cleared successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "warning",
                "message": "Cache clear completed with warnings",
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("\n" + "=" * 70)
    logger.info("Iniciando servidor con:")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Puerto: {PORT}")
    logger.info(f"  Ambiente: {ENVIRONMENT}")
    logger.info("=" * 70 + "\n")

    uvicorn.run(
        "api_server:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
