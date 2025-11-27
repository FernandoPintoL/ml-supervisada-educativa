"""
FastAPI Service para Predicciones ML
Plataforma Educativa

Expone endpoints para consumir modelos entrenados:
- /predict/risk
- /predict/career
- /predict/trend
- /predict/progress
- /health
"""

import logging
import pickle
import json
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd

# Importar modelos
from models.performance_predictor import PerformancePredictor
from models.career_recommender import CareerRecommender
from models.trend_predictor import TrendPredictor
from models.progress_analyzer import ProgressAnalyzer
from models.clustering_service import ClusteringService
from models.lstm_anomaly_detector import LSTMAnomalyDetector
from data.data_processor import DataProcessor
from shared.variable_schema import UNIFIED_SCHEMA, THRESHOLDS, VariableMapper

# ============================================================================
# CONFIGURACIÓN Y LOGGING
# ============================================================================

app = FastAPI(
    title="Plataforma Educativa - ML Prediction Service",
    version="1.0.0",
    description="API para predicciones ML de desempeño académico"
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================================
# MODELOS PYDANTIC (Request/Response)
# ============================================================================

class AnomalyDetectionRequest(BaseModel):
    """Request para detección de anomalías en series de calificaciones"""
    student_id: int = Field(..., description="ID del estudiante")
    grades: List[float] = Field(..., description="Lista de calificaciones (cronológico)")

    @validator('grades')
    def validate_grades(cls, v):
        if not v or len(v) < 2:
            raise ValueError('Se necesitan al menos 2 calificaciones')
        if not all(0 <= g <= 100 for g in v):
            raise ValueError('Las calificaciones deben estar entre 0 y 100')
        return v


class StudentFeatures(BaseModel):
    """Features de estudiante para predicción"""
    student_id: int = Field(..., description="ID del estudiante")
    promedio: float = Field(..., ge=0, le=100, description="Promedio actual (0-100)")
    asistencia: float = Field(default=80, ge=0, le=100, description="Porcentaje de asistencia")
    trabajos_entregados: int = Field(default=0, ge=0, description="Trabajos entregados")
    trabajos_totales: int = Field(default=1, ge=1, description="Total de trabajos asignados")
    varianza_calificaciones: float = Field(default=0, ge=0, description="Varianza de calificaciones")
    dias_desde_ultima_calificacion: int = Field(default=0, ge=0, description="Días desde última calificación")
    num_consultas_materiales: int = Field(default=0, ge=0, description="Número de consultas de materiales")

    @validator('trabajos_totales')
    def validate_trabajos(cls, v, values):
        if 'trabajos_entregados' in values:
            if values['trabajos_entregados'] > v:
                raise ValueError('trabajos_entregados no puede ser mayor que trabajos_totales')
        return v


class RiskPredictionResponse(BaseModel):
    """Respuesta de predicción de riesgo"""
    student_id: int
    score_riesgo: float = Field(..., ge=0, le=1)
    nivel_riesgo: str = Field(..., regex="^(alto|medio|bajo)$")
    confianza: float = Field(..., ge=0, le=1)
    caracteristicas_importancia: Dict[str, float]
    modelo_version: str
    timestamp: datetime
    metadata: Dict = {}


class CareerPredictionResponse(BaseModel):
    """Respuesta de predicción de carrera"""
    student_id: int
    carrera_nombre: str
    compatibilidad: float = Field(..., ge=0, le=1)
    ranking: int = Field(..., ge=1, le=3)
    confianza: float = Field(..., ge=0, le=1)
    modelo_version: str
    timestamp: datetime


class TrendPredictionResponse(BaseModel):
    """Respuesta de predicción de tendencia"""
    student_id: int
    tendencia: str = Field(..., regex="^(mejorando|estable|declinando|fluctuando)$")
    confianza: float = Field(..., ge=0, le=1)
    modelo_version: str
    timestamp: datetime


class ProgressPredictionResponse(BaseModel):
    """Respuesta de predicción de progreso"""
    student_id: int
    velocidad_aprendizaje: float
    nota_proyectada: float = Field(..., ge=0, le=100)
    confianza: float = Field(..., ge=0, le=1)
    modelo_version: str
    timestamp: datetime


class ClusteringPredictionResponse(BaseModel):
    """Respuesta de predicción de clustering"""
    student_id: int
    cluster_id: int = Field(..., ge=0, le=2)
    cluster_name: str
    expected_risk: str = Field(..., regex="^(alto|medio|bajo)$")
    cluster_probability: float = Field(..., ge=0, le=1)
    silhouette_score: float = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0, le=1)
    modelo_version: str
    timestamp: datetime


class AnomalyDetectionResponse(BaseModel):
    """Respuesta de detección de anomalías LSTM"""
    student_id: int
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0, le=1)
    anomaly_type: Optional[str] = Field(default=None, regex="^(spike_down|spike_up|volatility|drift)$")
    modelo_version: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    models_loaded: bool
    timestamp: datetime
    version: str = "1.0.0"


# ============================================================================
# CARGA DE MODELOS
# ============================================================================

MODELS = {
    'risk': None,
    'career': None,
    'trend': None,
    'progress': None,
    'clustering': None,
    'anomaly': None,
}

MODEL_VERSIONS = {
    'risk': 'v1.1-active',
    'career': 'v1.1-active',
    'trend': 'v1.1-active',
    'progress': 'v1.1-active',
    'clustering': 'v1.0-active',
    'anomaly': 'v1.0-active',
}


def load_models():
    """Cargar modelos entrenados al iniciar"""
    try:
        with open('trained_models/performance_predictor.pkl', 'rb') as f:
            MODELS['risk'] = pickle.load(f)
            logger.info("✓ Risk model loaded")

        with open('trained_models/career_recommender.pkl', 'rb') as f:
            MODELS['career'] = pickle.load(f)
            logger.info("✓ Career model loaded")

        with open('trained_models/trend_predictor.pkl', 'rb') as f:
            MODELS['trend'] = pickle.load(f)
            logger.info("✓ Trend model loaded")

        with open('trained_models/progress_analyzer.pkl', 'rb') as f:
            MODELS['progress'] = pickle.load(f)
            logger.info("✓ Progress model loaded")

        # Cargar modelo de clustering
        MODELS['clustering'] = ClusteringService()
        try:
            if MODELS['clustering'].load_model('trained_models/clustering_service.pkl'):
                logger.info("✓ Clustering model loaded")
            else:
                logger.warning("⚠ Clustering model not found. Using empty model.")
                MODELS['clustering'] = ClusteringService()
        except Exception as e:
            logger.warning(f"⚠ Could not load clustering model: {str(e)}. Using empty model.")
            MODELS['clustering'] = ClusteringService()

        # Cargar modelo de detección de anomalías LSTM
        MODELS['anomaly'] = LSTMAnomalyDetector()
        try:
            if MODELS['anomaly'].load_model('trained_models/lstm_anomaly_detector.pkl'):
                logger.info("✓ LSTM Anomaly Detector loaded")
            else:
                logger.warning("⚠ LSTM Anomaly Detector model not found. Using untrained model.")
                MODELS['anomaly'] = LSTMAnomalyDetector()
        except Exception as e:
            logger.warning(f"⚠ Could not load LSTM model: {str(e)}. Using untrained model.")
            MODELS['anomaly'] = LSTMAnomalyDetector()

        logger.info("All models loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Ejecutar al iniciar la aplicación"""
    load_models()


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar salud del servicio"""
    all_loaded = all(model is not None for model in MODELS.values())
    return HealthResponse(
        status="healthy" if all_loaded else "degraded",
        models_loaded=all_loaded,
        timestamp=datetime.now()
    )


@app.post("/predict/risk", response_model=RiskPredictionResponse)
async def predict_risk(features: StudentFeatures) -> Dict:
    """
    Predecir riesgo de bajo desempeño

    Returns:
        - score_riesgo: Score entre 0 y 1 (1 = máximo riesgo)
        - nivel_riesgo: Categoría (alto/medio/bajo)
        - confianza: Confianza de la predicción (probabilidad máxima)
        - caracteristicas_importancia: Features más importantes
    """
    try:
        # Validar modelo cargado
        if MODELS['risk'] is None:
            raise HTTPException(
                status_code=503,
                detail="Risk model not available. Service starting up."
            )

        # Validar entrada
        if features.promedio < 0 or features.promedio > 100:
            raise ValueError("Promedio debe estar entre 0-100")

        # Preprocesar features
        processed = DataProcessor.process_single_student(features.dict())

        # Predicción
        model = MODELS['risk']
        prediction_proba = model.predict_proba(processed)[0]

        # Score = probabilidad de "alto riesgo" (clase 0)
        risk_score = float(prediction_proba[0])
        risk_confidence = float(max(prediction_proba))

        # Determinar nivel basado en threshold
        if risk_score >= THRESHOLDS['risk']['high']:
            risk_level = 'alto'
        elif risk_score >= THRESHOLDS['risk']['medium_low']:
            risk_level = 'medio'
        else:
            risk_level = 'bajo'

        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(
                zip(
                    model.feature_names_,
                    [float(f) for f in model.feature_importances_]
                )
            )

        logger.info(
            f"Risk prediction for student {features.student_id}: "
            f"{risk_level} (score={risk_score:.3f}, conf={risk_confidence:.3f})"
        )

        return RiskPredictionResponse(
            student_id=features.student_id,
            score_riesgo=risk_score,
            nivel_riesgo=risk_level,
            confianza=risk_confidence,
            caracteristicas_importancia=feature_importance,
            modelo_version=MODEL_VERSIONS['risk'],
            timestamp=datetime.now(),
            metadata={
                'model_type': 'ensemble_rf_xgb',
                'cv_score': getattr(model, 'cv_score_', None),
                'n_features': len(feature_importance)
            }
        ).dict()

    except ValueError as e:
        logger.error(f"Validation error in predict/risk: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in predict/risk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/career", response_model=CareerPredictionResponse)
async def predict_career(features: StudentFeatures) -> Dict:
    """
    Predecir recomendación de carrera

    Returns:
        - carrera_nombre: Nombre de la carrera recomendada
        - compatibilidad: Score de compatibilidad (0-1)
        - ranking: Posición de la recomendación (1-3)
        - confianza: Confianza de la predicción
    """
    try:
        if MODELS['career'] is None:
            raise HTTPException(status_code=503, detail="Career model not available")

        # Preprocesar
        processed = DataProcessor.process_single_student(features.dict())

        # Predicción
        model = MODELS['career']
        prediction_proba = model.predict_proba(processed)[0]

        # Obtener carrera predicha
        predicted_class = np.argmax(prediction_proba)
        confidence = float(max(prediction_proba))

        # Mapear a nombre de carrera (asumiendo 8 carreras)
        CAREER_NAMES = [
            'Ingeniería en Sistemas',
            'Administración de Empresas',
            'Enfermería',
            'Derecho',
            'Psicología',
            'Educación',
            'Economía',
            'Contabilidad'
        ]
        career_name = CAREER_NAMES[predicted_class]

        logger.info(
            f"Career prediction for student {features.student_id}: "
            f"{career_name} (compatibility={confidence:.3f})"
        )

        return CareerPredictionResponse(
            student_id=features.student_id,
            carrera_nombre=career_name,
            compatibilidad=confidence,
            ranking=1,  # Top recommendation
            confianza=confidence,
            modelo_version=MODEL_VERSIONS['career'],
            timestamp=datetime.now()
        ).dict()

    except Exception as e:
        logger.error(f"Error in predict/career: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/trend", response_model=TrendPredictionResponse)
async def predict_trend(features: StudentFeatures) -> Dict:
    """
    Predecir tendencia de desempeño

    Returns:
        - tendencia: Tipo de tendencia (mejorando/estable/declinando/fluctuando)
        - confianza: Confianza de la predicción
    """
    try:
        if MODELS['trend'] is None:
            raise HTTPException(status_code=503, detail="Trend model not available")

        # Preprocesar
        processed = DataProcessor.process_single_student(features.dict())

        # Predicción
        model = MODELS['trend']
        prediction_proba = model.predict_proba(processed)[0]

        # Obtener tendencia predicha
        predicted_class = np.argmax(prediction_proba)
        confidence = float(max(prediction_proba))

        # Mapear a nombre de tendencia
        TREND_NAMES = ['mejorando', 'estable', 'declinando', 'fluctuando']
        trend = TREND_NAMES[predicted_class]

        logger.info(
            f"Trend prediction for student {features.student_id}: "
            f"{trend} (conf={confidence:.3f})"
        )

        return TrendPredictionResponse(
            student_id=features.student_id,
            tendencia=trend,
            confianza=confidence,
            modelo_version=MODEL_VERSIONS['trend'],
            timestamp=datetime.now()
        ).dict()

    except Exception as e:
        logger.error(f"Error in predict/trend: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/progress", response_model=ProgressPredictionResponse)
async def predict_progress(features: StudentFeatures) -> Dict:
    """
    Predecir progreso y velocidad de aprendizaje

    Returns:
        - velocidad_aprendizaje: Cambio proyectado de promedio
        - nota_proyectada: Calificación proyectada para próximo período
        - confianza: Confianza de la predicción
    """
    try:
        if MODELS['progress'] is None:
            raise HTTPException(status_code=503, detail="Progress model not available")

        # Preprocesar
        processed = DataProcessor.process_single_student(features.dict())

        # Predicción
        model = MODELS['progress']
        prediction = model.predict(processed)[0]

        # Calcular velocidad de aprendizaje (cambio por período)
        learning_velocity = prediction[0] if len(prediction) > 0 else 0
        projected_grade = float(features.promedio + learning_velocity)

        # Clamp a rango válido
        projected_grade = max(0, min(100, projected_grade))

        # Confianza basada en estabilidad
        confidence = min(1.0, 0.85 if abs(learning_velocity) < 5 else 0.65)

        logger.info(
            f"Progress prediction for student {features.student_id}: "
            f"velocity={learning_velocity:.2f}, projected={projected_grade:.1f}"
        )

        return ProgressPredictionResponse(
            student_id=features.student_id,
            velocidad_aprendizaje=float(learning_velocity),
            nota_proyectada=projected_grade,
            confianza=confidence,
            modelo_version=MODEL_VERSIONS['progress'],
            timestamp=datetime.now()
        ).dict()

    except Exception as e:
        logger.error(f"Error in predict/progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PREDICCIÓN BATCH (para sincronización)
# ============================================================================

class BatchPredictionRequest(BaseModel):
    """Request para predicción batch"""
    students: List[StudentFeatures]
    models: List[str] = Field(default=['risk'], description="Modelos a usar")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """
    Hacer predicciones para múltiples estudiantes

    Útil para sincronización periódica desde PHP
    """
    try:
        results = {
            'risk': [],
            'career': [],
            'trend': [],
            'progress': []
        }

        for features in request.students:
            if 'risk' in request.models:
                risk_result = await predict_risk(features)
                results['risk'].append(risk_result)

            if 'career' in request.models:
                career_result = await predict_career(features)
                results['career'].append(career_result)

            if 'trend' in request.models:
                trend_result = await predict_trend(features)
                results['trend'].append(trend_result)

            if 'progress' in request.models:
                progress_result = await predict_progress(features)
                results['progress'].append(progress_result)

        return {
            'success': True,
            'total_students': len(request.students),
            'results': results,
            'timestamp': datetime.now()
        }

    except Exception as e:
        logger.error(f"Error in predict/batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CLUSTERING
# ============================================================================

@app.post("/predict/cluster", response_model=ClusteringPredictionResponse)
async def predict_cluster(features: StudentFeatures) -> Dict:
    """
    Predecir cluster (grupo de estudiantes similar)

    Returns:
        - cluster_id: ID del cluster (0=bajo desempeño, 1=medio, 2=alto)
        - cluster_name: Nombre del cluster
        - expected_risk: Riesgo esperado según el cluster
        - cluster_probability: Probabilidad de pertenencia al cluster
        - silhouette_score: Score de calidad del cluster (-1 a 1)
        - confidence: Confianza general de la predicción
    """
    try:
        if MODELS['clustering'] is None:
            raise HTTPException(status_code=503, detail="Clustering model not available")

        # Preprocesar features
        processed = DataProcessor.process_single_student(features.dict())

        # Crear diccionario con valores
        feature_dict = {
            'promedio': features.promedio,
            'asistencia': features.asistencia,
            'trabajos_entregados': features.trabajos_entregados,
            'trabajos_totales': features.trabajos_totales,
            'varianza_calificaciones': features.varianza_calificaciones,
            'dias_desde_ultima_calificacion': features.dias_desde_ultima_calificacion,
            'num_consultas_materiales': features.num_consultas_materiales,
        }

        # Predicción de cluster
        model = MODELS['clustering']
        prediction = model.predict_cluster(feature_dict)

        if 'error' in prediction:
            raise HTTPException(status_code=500, detail=prediction['error'])

        logger.info(
            f"Cluster prediction for student {features.student_id}: "
            f"cluster={prediction['cluster_id']} "
            f"({prediction['cluster_name']}, "
            f"expected_risk={prediction['expected_risk']}, "
            f"conf={prediction['confidence']:.3f})"
        )

        return ClusteringPredictionResponse(
            student_id=features.student_id,
            cluster_id=prediction['cluster_id'],
            cluster_name=prediction['cluster_name'],
            expected_risk=prediction['expected_risk'],
            cluster_probability=prediction['cluster_probability'],
            silhouette_score=prediction['silhouette_score'],
            confidence=prediction['confidence'],
            modelo_version=MODEL_VERSIONS['clustering'],
            timestamp=datetime.now()
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict/cluster: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

@app.post("/detect/anomaly", response_model=AnomalyDetectionResponse)
async def detect_anomaly(request: AnomalyDetectionRequest) -> Dict:
    """
    Detectar anomalías en series de calificaciones usando LSTM

    Args:
        student_id: ID del estudiante
        grades: Lista de calificaciones (en orden cronológico)

    Returns:
        - is_anomaly: Si se detectó anomalía
        - anomaly_score: Score de anomalía (0-1)
        - anomaly_type: Tipo de anomalía detectada
    """
    try:
        if MODELS['anomaly'] is None:
            raise HTTPException(status_code=503, detail="Anomaly detector not available")

        # Detectar anomalía
        model = MODELS['anomaly']
        result = model.detect_anomaly(request.grades, request.student_id)

        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])

        logger.info(
            f"Anomaly detection for student {request.student_id}: "
            f"is_anomaly={result['is_anomaly']}, "
            f"type={result['anomaly_type']}, "
            f"score={result['anomaly_score']:.3f}"
        )

        return AnomalyDetectionResponse(
            student_id=request.student_id,
            is_anomaly=result['is_anomaly'],
            anomaly_score=result['anomaly_score'],
            anomaly_type=result['anomaly_type'],
            modelo_version=MODEL_VERSIONS['anomaly'],
            timestamp=datetime.now()
        ).dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect/anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
async def root():
    """Información del servicio"""
    return {
        'service': 'Plataforma Educativa - ML Prediction Service',
        'version': '1.0.0',
        'endpoints': [
            'GET /health',
            'POST /predict/risk',
            'POST /predict/career',
            'POST /predict/trend',
            'POST /predict/progress',
            'POST /predict/cluster',
            'POST /detect/anomaly',
            'POST /predict/batch'
        ],
        'docs': '/docs'
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
