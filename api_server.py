"""
FastAPI Prediction Server para Plataforma Educativa
Sirve predicciones ML entrenadas desde los modelos supervisados

Uso:
    uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Agregar ml_educativas al path
current_file = os.path.abspath(__file__)
ml_educativas_dir = os.path.dirname(current_file)
if ml_educativas_dir not in sys.path:
    sys.path.insert(0, ml_educativas_dir)

from shared.config import DEBUG, LOG_LEVEL, MODELS_DIR
from supervisado.data.data_processor import DataProcessor
from supervisado.models.performance_predictor import PerformancePredictor
from supervisado.models.career_recommender import CareerRecommender
from supervisado.models.trend_predictor import TrendPredictor, TREND_LABELS
from supervisado.models.progress_analyzer import ProgressAnalyzer
from cache.cache_manager import get_cache_manager, load_cached_dataset
from auth.sanctum_auth import (
    init_sanctum_auth,
    get_authenticator,
    SanctumRoleValidator,
    token_cache,
)

# Agregar directorio del agente al path
agente_dir = os.path.join(os.path.dirname(ml_educativas_dir), 'agente')
if agente_dir not in sys.path:
    sys.path.insert(0, agente_dir)

from recommendation_agent import get_agent

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    risk_level: str  # HIGH, MEDIUM, LOW
    risk_score: float  # 0-1
    confidence: float  # 0-1
    timestamp: str


class CareerPredictionResponse(BaseModel):
    """Respuesta de recomendación de carrera"""
    student_id: int
    top_3_careers: List[Dict[str, Any]]  # [{"career": "...", "compatibility": 0.85, "rank": 1}, ...]
    timestamp: str


class TrendPredictionResponse(BaseModel):
    """Respuesta de predicción de tendencia"""
    student_id: int
    trend: str  # improving, stable, declining, fluctuating
    confidence: float  # 0-1
    timestamp: str


class ProgressPredictionResponse(BaseModel):
    """Respuesta de predicción de progreso"""
    student_id: int
    projected_grade: float
    learning_velocity: float
    acceleration: float
    confidence: float  # 0-1
    timestamp: str


class HealthCheckResponse(BaseModel):
    """Respuesta de verificación de salud"""
    status: str  # "healthy" or "unhealthy"
    models_loaded: Dict[str, bool]
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


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Plataforma Educativa - ML Prediction API",
    description="Servidor de predicciones ML para análisis educativo",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODEL LOADER & MANAGER
# ============================================================

class ModelManager:
    """Gestiona carga y predicciones de modelos ML"""

    def __init__(self):
        self.models = {}
        self.processor = None
        self.is_ready = False
        self.cache_manager = get_cache_manager()
        self.cached_data = None
        self.cached_features = None
        self.cache_loaded = False

    def load_models(self):
        """Cargar todos los modelos entrenados"""
        logger.info("Iniciando carga de modelos...")

        try:
            # Cargar Performance Predictor (Risk)
            logger.info("  Cargando PerformancePredictor...")
            risk_model_path = os.path.join(MODELS_DIR, 'PerformancePredictor_model.pkl')
            if os.path.exists(risk_model_path):
                self.models['risk'] = PerformancePredictor()
                self.models['risk'].load(risk_model_path)
                logger.info("    ✓ PerformancePredictor cargado")
            else:
                logger.warning(f"    ✗ Archivo no encontrado: {risk_model_path}")

            # Cargar Career Recommender
            logger.info("  Cargando CareerRecommender...")
            career_model_path = os.path.join(MODELS_DIR, 'CareerRecommender_model.pkl')
            if os.path.exists(career_model_path):
                self.models['career'] = CareerRecommender()
                self.models['career'].load(career_model_path)
                logger.info("    ✓ CareerRecommender cargado")
            else:
                logger.warning(f"    ✗ Archivo no encontrado: {career_model_path}")

            # Cargar Trend Predictor
            logger.info("  Cargando TrendPredictor...")
            trend_model_path = os.path.join(MODELS_DIR, 'TrendPredictor_model.pkl')
            if os.path.exists(trend_model_path):
                self.models['trend'] = TrendPredictor()
                self.models['trend'].load(trend_model_path)
                logger.info("    ✓ TrendPredictor cargado")
            else:
                logger.warning(f"    ✗ Archivo no encontrado: {trend_model_path}")

            # Cargar Progress Analyzer
            logger.info("  Cargando ProgressAnalyzer...")
            progress_model_path = os.path.join(MODELS_DIR, 'ProgressAnalyzer_model.pkl')
            if os.path.exists(progress_model_path):
                self.models['progress'] = ProgressAnalyzer()
                self.models['progress'].load(progress_model_path)
                logger.info("    ✓ ProgressAnalyzer cargado")
            else:
                logger.warning(f"    ✗ Archivo no encontrado: {progress_model_path}")

            # Inicializar processor
            self.processor = DataProcessor(scaler_type="standard")

            self.is_ready = len(self.models) > 0
            logger.info(f"✓ Cargados {len(self.models)} modelos")

            # Intentar cargar dataset desde caché
            self.load_cached_dataset()

            return self.is_ready

        except Exception as e:
            logger.error(f"✗ Error cargando modelos: {str(e)}", exc_info=True)
            return False

    def load_cached_dataset(self):
        """Cargar dataset de caché para acelerar predicciones"""
        try:
            logger.info("\n[Cache] Intentando cargar dataset desde caché...")
            cache_info = self.cache_manager.get_cache_info()

            if not cache_info['exists']:
                logger.info("  ⚠ No hay caché disponible - será cargado desde BD en cada predicción")
                return False

            # Intentar cargar el dataset
            result = self.cache_manager.load_dataset()
            if result:
                self.cached_data, self.cached_features = result
                self.cache_loaded = True
                logger.info(f"  ✓ Dataset cargado desde caché")
                logger.info(f"    - Registros: {cache_info['num_records']}")
                logger.info(f"    - Features: {cache_info['num_features']}")
                logger.info(f"    - Tamaño: {cache_info['size_mb']}MB")
                logger.info(f"    - Timestamp: {cache_info['timestamp']}")
                return True
            else:
                logger.warning("  ✗ Error al cargar dataset desde caché")
                return False

        except Exception as e:
            logger.warning(f"  ⚠ Error cargando caché: {str(e)}")
            return False

    def get_cached_dataset_info(self) -> Dict:
        """Obtener información del dataset cacheado"""
        return self.cache_manager.get_cache_info()

    def prepare_features(self, student_data: StudentData) -> np.ndarray:
        """Preparar características del estudiante como array numpy"""
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
# DEPENDENCIAS DE AUTENTICACIÓN
# ============================================================

async def verify_sanctum_token(authorization: str = Header(None)) -> Dict:
    """
    Dependencia para verificar token de Sanctum

    Valida que el token sea válido y retorna información del usuario
    """
    if not authorization:
        logger.warning("Request sin token Authorization")
        raise HTTPException(status_code=401, detail="Token requerido")

    try:
        # Extraer token del header "Bearer <token>"
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Formato de token inválido")

        token = authorization[7:]  # Remover "Bearer "

        # Verificar en cache primero
        cached = token_cache.get(token)
        if cached:
            logger.debug(f"Token encontrado en cache")
            return cached

        # Validar token con Sanctum
        authenticator = get_authenticator()
        token_data = authenticator.validate_sanctum_token(token)

        if not token_data or not token_data.get("valid"):
            logger.warning(f"Token inválido")
            raise HTTPException(status_code=401, detail="Token inválido")

        # Guardar en cache
        token_cache.set(token, token_data)

        logger.info(f"Token validado: {token_data.get('token_id')}")
        return token_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validando token: {str(e)}")
        raise HTTPException(status_code=401, detail="Error validando token")


def require_role(required_role: str):
    """
    Crear dependencia que requiere un rol específico

    Uso: @app.post("/endpoint", dependencies=[Depends(require_role("admin"))])
    """
    async def check_role(token_data: Dict = Depends(verify_sanctum_token)) -> Dict:
        # Verificar que el usuario tiene el rol requerido
        user_info = SanctumRoleValidator.get_user_info(token_data)

        if not SanctumRoleValidator.has_role(user_info, required_role):
            logger.warning(
                f"Usuario {user_info.get('user_id')} no tiene rol {required_role}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Rol requerido: {required_role}"
            )

        return token_data

    return check_role


def require_permission(required_permission: str):
    """
    Crear dependencia que requiere un permiso específico

    Uso: @app.post("/endpoint", dependencies=[Depends(require_permission("predict:all"))])
    """
    async def check_permission(token_data: Dict = Depends(verify_sanctum_token)) -> Dict:
        user_info = SanctumRoleValidator.get_user_info(token_data)

        if not SanctumRoleValidator.has_permission(user_info, required_permission):
            logger.warning(
                f"Usuario {user_info.get('user_id')} no tiene permiso {required_permission}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Permiso requerido: {required_permission}"
            )

        return token_data

    return check_permission


# ============================================================
# STARTUP & SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio - cargar modelos y autenticación"""
    logger.info("=" * 60)
    logger.info("INICIANDO SERVIDOR API ML")
    logger.info("=" * 60)

    # Inicializar autenticación Sanctum
    try:
        laravel_key = os.getenv("LARAVEL_APP_KEY", os.getenv("APP_KEY"))
        if laravel_key:
            init_sanctum_auth(laravel_key)
            logger.info("✓ Autenticación Sanctum inicializada")
        else:
            logger.warning("⚠ LARAVEL_APP_KEY no encontrada - API sin autenticación")
    except Exception as e:
        logger.error(f"Error inicializando Sanctum: {str(e)}")

    # Cargar modelos
    model_manager.load_models()
    if model_manager.is_ready:
        logger.info("✓ Servidor listo para recibir predicciones")
    else:
        logger.warning("⚠ Servidor iniciado pero sin modelos cargados")


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
        models_loaded=models_loaded,
        timestamp=datetime.now().isoformat()
    )


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================

@app.post("/predict/risk", response_model=RiskPredictionResponse)
async def predict_risk(
    student_data: StudentData,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """
    Predecir nivel de riesgo del estudiante

    **Requiere:** Token válido de Sanctum

    Risk Levels:
    - HIGH (0.7-1.0): Intervención inmediata recomendada
    - MEDIUM (0.4-0.7): Monitoreo cercano
    - LOW (0.0-0.4): Desempeño aceptable
    """
    logger.info(f"Predicción de riesgo solicitada (token: {token_data.get('token_id')})")

    if 'risk' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Risk model not loaded")

    try:
        # Preparar features
        X = model_manager.prepare_features(student_data)

        # Obtener predicción
        risk_score = model_manager.models['risk'].predict(X)[0]

        # Clasificar nivel de riesgo
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Confianza (asumir del modelo)
        confidence = abs(risk_score - 0.5) * 2  # Simplificado
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


@app.post("/predict/career", response_model=CareerPredictionResponse)
async def predict_career(
    student_data: StudentData,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """
    Recomendar carreras basado en desempeño académico
    Retorna top 3 carreras con compatibilidad

    **Requiere:** Token válido de Sanctum
    """
    logger.info(f"Predicción de carrera solicitada (token: {token_data.get('token_id')})")

    if 'career' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Career model not loaded")

    try:
        # Preparar features
        X = model_manager.prepare_features(student_data)

        # Obtener predicción
        predictions = model_manager.models['career'].predict(X)[0]
        probabilities = model_manager.models['career'].predict_proba(X)[0]

        # Obtener top 3
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


@app.post("/predict/trend", response_model=TrendPredictionResponse)
async def predict_trend(
    student_data: StudentData,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """
    Predecir tendencia de aprendizaje

    **Requiere:** Token válido de Sanctum

    Trends:
    - improving: Calificaciones aumentando
    - stable: Desempeño consistente
    - declining: Calificaciones decreciendo
    - fluctuating: Rendimiento variable
    """
    logger.info(f"Predicción de tendencia solicitada (token: {token_data.get('token_id')})")

    if 'trend' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Trend model not loaded")

    try:
        # Preparar features
        X = model_manager.prepare_features(student_data)

        # Obtener predicción
        trend_idx = model_manager.models['trend'].predict(X)[0]
        probabilities = model_manager.models['trend'].predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        # Mapear a label
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


@app.post("/predict/progress", response_model=ProgressPredictionResponse)
async def predict_progress(
    student_data: StudentData,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """
    Predecir progreso académico
    Retorna proyección de calificación, velocidad de aprendizaje y aceleración

    **Requiere:** Token válido de Sanctum
    """
    logger.info(f"Predicción de progreso solicitada (token: {token_data.get('token_id')})")

    if 'progress' not in model_manager.models:
        raise HTTPException(status_code=503, detail="Progress model not loaded")

    try:
        # Preparar features
        X = model_manager.prepare_features(student_data)

        # Obtener predicción
        projected_grade = model_manager.models['progress'].predict(X)[0]

        # Calcular velocidad de aprendizaje
        current_grade = student_data.promedio_calificaciones
        learning_velocity = (projected_grade - current_grade) / max(student_data.num_calificaciones, 1)

        # Calcular aceleración (cambio en velocidad)
        acceleration = learning_velocity * (student_data.varianza_calificaciones / 10)

        # Confianza basada en número de calificaciones
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
# BATCH PREDICTION ENDPOINT
# ============================================================

@app.post("/predict/batch")
async def predict_batch(students: List[StudentData]):
    """
    Predecir para múltiples estudiantes en una sola llamada
    Retorna todas las predicciones combinadas
    """
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
                    student_result['risk'] = {
                        'level': risk_level,
                        'score': float(risk_score)
                    }
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
# EDUCATIONAL RECOMMENDATIONS ENDPOINT
# ============================================================

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    token_data: Dict = Depends(verify_sanctum_token)
):
    """
    Generar recomendaciones educativas personalizadas usando Agente IA

    **Requiere:** Token válido de Sanctum

    El agente analiza:
    - Predicciones de ML (riesgo, progreso, tendencia)
    - Datos académicos del estudiante
    - Patrones de desempeño

    Retorna:
    - Tipo de recomendación (intervention, tutoring, enrichment, study_resource)
    - Urgencia (immediate, normal, preventive)
    - Acciones específicas
    - Recursos recomendados
    - Indicadores de éxito
    """
    try:
        logger.info(
            f"Generating recommendations for student {request.student_id} "
            f"(token: {token_data.get('token_id')})"
        )

        # Obtener agente inteligente
        agent = get_agent()

        # Preparar datos del estudiante
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

        # Preparar predicciones de ML
        predictions = {
            'risk_score': request.risk_score,
            'risk_level': request.risk_level,
            'projected_grade': request.projected_grade,
            'trend': request.trend,
            'confidence': request.confidence,
        }

        # Generar recomendaciones con agente
        logger.info("Calling AI agent to generate recommendations...")
        agent_response = agent.generate_recommendations(student_data, predictions)

        # Formatear respuesta
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

        logger.info(f"Recommendations generated successfully for student {request.student_id}")
        return response

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CACHE MANAGEMENT ENDPOINT
# ============================================================

@app.post("/cache/refresh")
async def refresh_cache():
    """Refrescar/recargar el caché de datasets"""
    try:
        logger.info("=" * 60)
        logger.info("REFRESHING CACHE")
        logger.info("=" * 60)

        # Recargar el caché
        success = model_manager.load_cached_dataset()

        if success:
            cache_info = model_manager.get_cached_dataset_info()
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
        raise HTTPException(status_code=500, detail=f"Error refreshing cache: {str(e)}")


@app.get("/cache/info")
async def get_cache_info():
    """Obtener información del caché actual"""
    try:
        cache_info = model_manager.get_cached_dataset_info()
        return {
            "status": "success",
            "cache": cache_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting cache info: {str(e)}")


@app.delete("/cache/clear")
async def clear_cache():
    """Limpiar el caché completamente"""
    try:
        logger.info("=" * 60)
        logger.info("CLEARING CACHE")
        logger.info("=" * 60)

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
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    """Endpoint raíz con información del servidor"""
    return {
        "name": "Plataforma Educativa - ML API",
        "version": "1.0.0",
        "status": "healthy" if model_manager.is_ready else "models_loading",
        "models_loaded": len(model_manager.models),
        "endpoints": {
            "health": "/health",
            "predict_risk": "/predict/risk",
            "predict_career": "/predict/career",
            "predict_trend": "/predict/trend",
            "predict_progress": "/predict/progress",
            "predict_batch": "/predict/batch",
            "cache_info": "/cache/info",
            "cache_refresh": "/cache/refresh",
            "cache_clear": "/cache/clear",
            "docs": "/docs",
            "redoc": "/redoc",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
