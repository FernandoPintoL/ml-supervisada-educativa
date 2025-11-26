#!/usr/bin/env python
"""
Script Simplificado de Entrenamiento de Modelos ML
INDEPENDIENTE - Se ejecuta completamente desde supervisado/
Se conecta directamente a la BD de Laravel
"""

import sys
import os
import logging
import pickle
import json
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de BD
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_DATABASE', 'educativa')
DB_USER = os.getenv('DB_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '1234')

MODELS_DIR = Path(__file__).parent / 'trained_models'
MODELS_DIR.mkdir(exist_ok=True)


class DBConnection:
    """Conexión a la base de datos de Laravel"""

    def __init__(self):
        self.conn = None

    def connect(self) -> bool:
        """Conectar a la BD"""
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info(f"[OK] Conectado a BD: {DB_NAME}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Error conectando a BD: {e}")
            return False

    def execute_query(self, query: str) -> list:
        """Ejecutar query y retornar resultados"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"[ERROR] Error en query: {e}")
            return []

    def close(self):
        """Cerrar conexión"""
        if self.conn:
            self.conn.close()


class MLTrainer:
    """Entrena modelos de ML con datos de la BD de Laravel"""

    def __init__(self):
        self.db = DBConnection()
        self.models = {}

    def cargar_datos(self) -> pd.DataFrame:
        """Cargar datos de la BD de Laravel"""
        logger.info("[*] Cargando datos de la base de datos...")

        if not self.db.connect():
            return None

        # Query para obtener datos de entrenamiento
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
            ra.promedio as promedio_rendimiento
        FROM users u
        LEFT JOIN trabajos t ON u.id = t.estudiante_id
        LEFT JOIN calificaciones c ON t.id = c.trabajo_id
        LEFT JOIN rendimiento_academico ra ON u.id = ra.estudiante_id
        WHERE u.tipo_usuario = 'estudiante'
        GROUP BY u.id, u.desempeño_promedio, u.asistencia_porcentaje,
                 u.participacion_porcentaje, u.tareas_completadas,
                 u.tareas_pendientes, u.actividad_hoy, ra.promedio
        LIMIT 100
        """

        results = self.db.execute_query(query)

        if not results:
            logger.error("[ERROR] No se encontraron datos de estudiantes")
            return None

        df = pd.DataFrame(results)
        logger.info(f"[OK] Datos cargados: {len(df)} estudiantes")
        logger.info(f"  Características: {list(df.columns)}")

        return df

    def entrenar_performance_predictor(self) -> bool:
        """Entrenar modelo de predicción de calificaciones"""
        logger.info("\n[MODEL] Entrenando Performance Predictor...")

        df = self.cargar_datos()
        if df is None or df.empty:
            logger.error("[ERROR] Sin datos para entrenar")
            return False

        # Features y target
        features = [
            'desempeño_promedio', 'asistencia_porcentaje',
            'participacion_porcentaje', 'tareas_completadas',
            'tareas_pendientes', 'actividad_hoy'
        ]

        # Filtrar solo filas con datos completos
        df_clean = df[features + ['promedio_calificaciones']].dropna()

        if df_clean.empty:
            logger.error("[ERROR] Sin datos limpios después del filtrado")
            return False

        X = df_clean[features]
        y = df_clean['promedio_calificaciones']

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Entrenar modelo
        logger.info("  Entrenando Random Forest...")
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Evaluar
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = model.score(X_test, y_test)

        logger.info(f"[OK] Modelo entrenado:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²: {r2:.4f}")

        # Guardar modelo
        model_path = MODELS_DIR / 'PerformancePredictor_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Modelo guardado: {model_path}")

        self.models['performance'] = model
        return True

    def entrenar_career_recommender(self) -> bool:
        """Entrenar modelo de recomendación de carreras"""
        logger.info("\n[MODEL] Entrenando Career Recommender...")

        df = self.cargar_datos()
        if df is None or df.empty:
            logger.error("[ERROR] Sin datos para entrenar")
            return False

        features = [
            'desempeño_promedio', 'asistencia_porcentaje',
            'participacion_porcentaje', 'promedio_rendimiento'
        ]

        df_clean = df[features].dropna()

        if df_clean.empty:
            logger.error("[ERROR] Sin datos limpios")
            return False

        X = df_clean[features]
        # Crear clasificación simple: Alto rendimiento (>70) vs Bajo
        y = (X['desempeño_promedio'] > 70).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("  Entrenando Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"[OK] Modelo entrenado:")
        logger.info(f"  Accuracy: {accuracy:.4f}")

        # Guardar
        model_path = MODELS_DIR / 'CareerRecommender_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Modelo guardado: {model_path}")

        self.models['career'] = model
        return True

    def entrenar_trend_predictor(self) -> bool:
        """Entrenar modelo de predicción de tendencias"""
        logger.info("\n[MODEL] Entrenando Trend Predictor...")

        df = self.cargar_datos()
        if df is None or df.empty:
            logger.error("[ERROR] Sin datos")
            return False

        # Datos similares para simplificar
        features = ['asistencia_porcentaje', 'participacion_porcentaje', 'actividad_hoy']
        df_clean = df[features + ['desempeño_promedio']].dropna()

        if df_clean.empty:
            logger.error("[ERROR] Sin datos limpios")
            return False

        X = df_clean[features]
        # Tendencia: mejorando si desempeño > 60, empeorando si < 40, estable en medio
        y = pd.cut(df_clean['desempeño_promedio'],
                   bins=[0, 40, 60, 100],
                   labels=[0, 1, 2]).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("  Entrenando modelo...")
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        logger.info(f"[OK] Modelo entrenado:")
        logger.info(f"  Accuracy: {accuracy:.4f}")

        model_path = MODELS_DIR / 'TrendPredictor_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Modelo guardado: {model_path}")

        self.models['trend'] = model
        return True

    def entrenar_progress_analyzer(self) -> bool:
        """Entrenar modelo de análisis de progreso"""
        logger.info("\n[MODEL] Entrenando Progress Analyzer...")

        df = self.cargar_datos()
        if df is None or df.empty:
            logger.error("[ERROR] Sin datos")
            return False

        features = ['tareas_completadas', 'tareas_pendientes', 'actividad_hoy']
        df_clean = df[features + ['desempeño_promedio']].dropna()

        if df_clean.empty:
            logger.error("[ERROR] Sin datos limpios")
            return False

        X = df_clean[features]
        y = df_clean['desempeño_promedio']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("  Entrenando modelo...")
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)

        logger.info(f"[OK] Modelo entrenado:")
        logger.info(f"  R²: {r2:.4f}")

        model_path = MODELS_DIR / 'ProgressAnalyzer_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Modelo guardado: {model_path}")

        self.models['progress'] = model
        return True

    def registrar_entrenamientos(self, resultados: dict) -> None:
        """Guardar registro de entrenamientos en JSON"""
        log_path = MODELS_DIR / 'training_log.json'

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'modelos': resultados,
            'directorio': str(MODELS_DIR),
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Log guardado en: {log_path}")

    def entrenar_todos(self) -> bool:
        """Entrenar todos los modelos"""
        logger.info("=" * 70)
        logger.info("INICIANDO ENTRENAMIENTO DE MODELOS ML")
        logger.info("=" * 70)

        resultados = {
            'Performance Predictor': self.entrenar_performance_predictor(),
            'Career Recommender': self.entrenar_career_recommender(),
            'Trend Predictor': self.entrenar_trend_predictor(),
            'Progress Analyzer': self.entrenar_progress_analyzer(),
        }

        logger.info("\n" + "=" * 70)
        logger.info("RESUMEN DE ENTRENAMIENTOS")
        logger.info("=" * 70)

        for modelo, exito in resultados.items():
            estado = "[OK]" if exito else "[ERROR]"
            logger.info(f"{estado} - {modelo}")

        logger.info("=" * 70)
        logger.info(f"Modelos guardados en: {MODELS_DIR}")

        # Registrar entrenamientos
        self.registrar_entrenamientos(resultados)

        self.db.close()

        return all(resultados.values())


def main():
    """Función principal"""
    try:
        trainer = MLTrainer()
        exito = trainer.entrenar_todos()

        if exito:
            logger.info("\n[SUCCESS] Todos los modelos entrenados exitosamente!")
            logger.info(f"Modelos guardados en: {MODELS_DIR}")
            logger.info("\nArchivos generados:")
            for f in MODELS_DIR.glob("*.pkl"):
                logger.info(f"  - {f.name}")
            return 0
        else:
            logger.error("\n[FAILED] Error en el entrenamiento")
            return 1
    except Exception as e:
        logger.error(f"[ERROR] Excepcion no manejada: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
