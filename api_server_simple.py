#!/usr/bin/env python
"""
API Server Simplificado para Predicciones ML
Carga y usa los modelos entrenados con train_models_simple.py
COMPLETAMENTE INDEPENDIENTE - TODO EN supervisado/
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_DATABASE', 'educativa')
DB_USER = os.getenv('DB_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '1234')

MODELS_DIR = Path(__file__).parent / 'trained_models'

# Crear app FastAPI
app = FastAPI(
    title="API Predicciones ML - Supervisada",
    description="API para predicciones de desempeño académico",
    version="1.0.0"
)

# Models storage
MODELS = {}


class PredictionRequest(BaseModel):
    """Solicitud de predicción"""
    student_id: int


class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    student_id: int
    prediction: float
    confidence: Optional[float] = None
    model_used: str


class QuestionDifficultyRequest(BaseModel):
    """Solicitud para predecir dificultad de pregunta"""
    enunciado: str
    tipo: str  # opcion_multiple, verdadero_falso, respuesta_corta
    nivel_bloom: str  # remember, understand, apply, analyze, evaluate, create
    curso_id: Optional[int] = None
    longitud_opciones: Optional[int] = None


class QuestionDifficultyResponse(BaseModel):
    """Respuesta con predicción de dificultad"""
    dificultad_predicha: float  # 0.0 - 1.0
    confianza: float  # 0.0 - 1.0
    clasificacion: str  # muy_facil, facil, media, dificil, muy_dificil
    razonamiento: str  # Explicación breve


class DBConnection:
    """Conexión a BD"""

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


def load_models():
    """Cargar modelos entrenados"""
    global MODELS

    logger.info("\n[*] Cargando modelos entrenados...")

    model_files = {
        'performance': 'PerformancePredictor_model.pkl',
        'career': 'CareerRecommender_model.pkl',
        'trend': 'TrendPredictor_model.pkl',
        'progress': 'ProgressAnalyzer_model.pkl',
    }

    for key, filename in model_files.items():
        model_path = MODELS_DIR / filename

        if not model_path.exists():
            logger.warning(f"[WARNING] Modelo no encontrado: {filename}")
            MODELS[key] = None
            continue

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            MODELS[key] = model
            logger.info(f"[OK] {key.upper()} cargado: {filename}")
        except Exception as e:
            logger.error(f"[ERROR] Error cargando {key}: {e}")
            MODELS[key] = None

    logger.info(f"[OK] Cargados {sum(1 for m in MODELS.values() if m)} modelos\n")


@app.on_event("startup")
async def startup():
    """Evento de inicio"""
    logger.info("=" * 70)
    logger.info("INICIANDO SERVIDOR API - PREDICCIONES ML")
    logger.info("=" * 70)
    logger.info(f"Directorio de modelos: {MODELS_DIR}")
    logger.info(f"Base de datos: {DB_NAME}@{DB_HOST}")
    load_models()
    logger.info("=" * 70)
    logger.info("[OK] Servidor listo para recibir predicciones")
    logger.info("=" * 70 + "\n")


@app.get("/health", tags=["Health"])
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "performance": MODELS.get('performance') is not None,
            "career": MODELS.get('career') is not None,
            "trend": MODELS.get('trend') is not None,
            "progress": MODELS.get('progress') is not None,
        }
    }


@app.post("/predict/performance", response_model=PredictionResponse, tags=["Predictions"])
async def predict_performance(request: PredictionRequest):
    """Predecir calificación de desempeño"""

    if MODELS['performance'] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Obtener datos del estudiante
    student_data = DBConnection.get_student_data(request.student_id)
    if not student_data:
        raise HTTPException(status_code=404, detail=f"Estudiante {request.student_id} no encontrado")

    try:
        # Preparar features en el orden correcto
        features = [
            student_data['desempeño_promedio'],
            student_data['asistencia_porcentaje'],
            student_data['participacion_porcentaje'],
            student_data['tareas_completadas'],
            student_data['tareas_pendientes'],
            student_data['actividad_hoy']
        ]

        # Hacer predicción
        prediction = MODELS['performance'].predict([features])[0]
        prediction = float(max(0, min(100, prediction)))  # Limitar entre 0-100

        logger.info(f"[PREDICT] Performance - Student: {request.student_id}, Prediction: {prediction:.2f}")

        return PredictionResponse(
            student_id=request.student_id,
            prediction=prediction,
            confidence=0.95,  # Basado en R² del modelo
            model_used="PerformancePredictor"
        )

    except Exception as e:
        logger.error(f"[ERROR] Predicción fallida: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/career", response_model=PredictionResponse, tags=["Predictions"])
async def predict_career(request: PredictionRequest):
    """Predecir recomendación de carrera"""

    if MODELS['career'] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    student_data = DBConnection.get_student_data(request.student_id)
    if not student_data:
        raise HTTPException(status_code=404, detail=f"Estudiante {request.student_id} no encontrado")

    try:
        features = [
            student_data['desempeño_promedio'],
            student_data['asistencia_porcentaje'],
            student_data['participacion_porcentaje'],
            student_data['promedio_rendimiento']
        ]

        prediction = MODELS['career'].predict([features])[0]

        careers = {
            0: "Carreras Técnicas",
            1: "Carreras Profesionales"
        }

        logger.info(f"[PREDICT] Career - Student: {request.student_id}, Recommendation: {careers.get(prediction, 'Unknown')}")

        return PredictionResponse(
            student_id=request.student_id,
            prediction=float(prediction),
            confidence=1.0,  # Accuracy perfecta
            model_used="CareerRecommender"
        )

    except Exception as e:
        logger.error(f"[ERROR] Predicción fallida: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/trend", response_model=PredictionResponse, tags=["Predictions"])
async def predict_trend(request: PredictionRequest):
    """Predecir tendencia de desempeño"""

    if MODELS['trend'] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    student_data = DBConnection.get_student_data(request.student_id)
    if not student_data:
        raise HTTPException(status_code=404, detail=f"Estudiante {request.student_id} no encontrado")

    try:
        features = [
            student_data['asistencia_porcentaje'],
            student_data['participacion_porcentaje'],
            student_data['actividad_hoy']
        ]

        prediction = MODELS['trend'].predict([features])[0]

        trends = {
            0: "Empeorando",
            1: "Estable",
            2: "Mejorando"
        }

        logger.info(f"[PREDICT] Trend - Student: {request.student_id}, Trend: {trends.get(prediction, 'Unknown')}")

        return PredictionResponse(
            student_id=request.student_id,
            prediction=float(prediction),
            confidence=0.90,
            model_used="TrendPredictor"
        )

    except Exception as e:
        logger.error(f"[ERROR] Predicción fallida: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/progress", response_model=PredictionResponse, tags=["Predictions"])
async def predict_progress(request: PredictionRequest):
    """Predecir análisis de progreso"""

    if MODELS['progress'] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    student_data = DBConnection.get_student_data(request.student_id)
    if not student_data:
        raise HTTPException(status_code=404, detail=f"Estudiante {request.student_id} no encontrado")

    try:
        features = [
            student_data['tareas_completadas'],
            student_data['tareas_pendientes'],
            student_data['actividad_hoy']
        ]

        prediction = MODELS['progress'].predict([features])[0]
        prediction = float(max(0, min(100, prediction)))  # Limitar entre 0-100

        logger.info(f"[PREDICT] Progress - Student: {request.student_id}, Progress: {prediction:.2f}")

        return PredictionResponse(
            student_id=request.student_id,
            prediction=prediction,
            confidence=0.91,
            model_used="ProgressAnalyzer"
        )

    except Exception as e:
        logger.error(f"[ERROR] Predicción fallida: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/question-difficulty", response_model=QuestionDifficultyResponse, tags=["Predictions"])
async def predict_question_difficulty(request: QuestionDifficultyRequest):
    """
    Predecir la dificultad de una pregunta educativa

    Basado en:
    - Complejidad del enunciado (longitud, palabras clave)
    - Nivel Bloom
    - Tipo de pregunta
    - Contexto del curso (si se proporciona)

    Retorna:
    - dificultad_predicha: 0.0 (muy fácil) a 1.0 (muy difícil)
    - confianza: 0.0 a 1.0
    - clasificacion: categoría de dificultad
    """

    try:
        # Extraer características del enunciado
        longitud_enunciado = len(request.enunciado.strip())
        num_palabras = len(request.enunciado.split())

        # Mapear nivel Bloom a numérico (0-5)
        bloom_mapping = {
            'remember': 0,
            'understand': 1,
            'apply': 2,
            'analyze': 3,
            'evaluate': 4,
            'create': 5
        }
        bloom_score = bloom_mapping.get(request.nivel_bloom.lower(), 2)

        # Mapear tipo de pregunta a numérico
        tipo_mapping = {
            'verdadero_falso': 0,
            'opcion_multiple': 1,
            'respuesta_corta': 2,
            'respuesta_larga': 3
        }
        tipo_score = tipo_mapping.get(request.tipo.lower(), 1)

        # Calcular dificultad basada en características
        # Fórmula: combinar Bloom (40%) + Longitud (30%) + Tipo (30%)

        # Bloom contribuye directamente (0-1 scale: 5/5 = 1.0)
        bloom_difficulty = bloom_score / 5.0

        # Longitud: preguntas muy cortas o muy largas son más difíciles
        # Óptimo: 50-150 caracteres
        if longitud_enunciado < 20 or longitud_enunciado > 300:
            longitud_difficulty = 0.8
        elif longitud_enunciado < 50 or longitud_enunciado > 200:
            longitud_difficulty = 0.6
        else:
            longitud_difficulty = 0.4

        # Tipo: respuesta larga > respuesta corta > opción múltiple > V/F
        tipo_difficulty = tipo_score / 3.0

        # Combinar características
        dificultad_predicha = (
            bloom_difficulty * 0.40 +
            longitud_difficulty * 0.30 +
            tipo_difficulty * 0.30
        )

        # Limitar entre 0 y 1
        dificultad_predicha = max(0.0, min(1.0, dificultad_predicha))

        # Clasificar dificultad
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

        # Generar razonamiento
        razones = []
        if bloom_score >= 4:
            razones.append(f"Nivel Bloom alto ({request.nivel_bloom})")
        if longitud_enunciado > 200:
            razones.append("Enunciado extenso")
        if tipo_score > 1:
            razones.append(f"Tipo de pregunta requiere más análisis ({request.tipo})")

        razonamiento = "; ".join(razones) if razones else "Dificultad promedio basada en características"

        # Confianza basada en consistencia de características
        confianza = 0.75 + (0.25 * abs(0.5 - dificultad_predicha))  # Mayor cuando está en extremos

        logger.info(
            f"[PREDICT] Dificultad Pregunta - "
            f"Dificultad: {dificultad_predicha:.3f}, "
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


@app.get("/", tags=["Info"])
async def root():
    """Información del servidor"""
    return {
        "name": "API Predicciones ML - Supervisada",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "performance": "POST /predict/performance",
            "career": "POST /predict/career",
            "trend": "POST /predict/trend",
            "progress": "POST /predict/progress",
            "question_difficulty": "POST /predict/question-difficulty",
        },
        "documentation": {
            "swagger": "http://localhost:8001/docs",
            "redoc": "http://localhost:8001/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
