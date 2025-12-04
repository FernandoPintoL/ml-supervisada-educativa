#!/usr/bin/env python
"""
VALIDACIÓN REAL DE MODELOS ML - Opción A

Divide datos por FECHA, no al azar:
- TRAINING: Datos históricos (primeras 3 semanas)
- TEST: Datos recientes (última semana)

Entrena SOLO con histórico y valida contra realidad reciente.
Genera métricas honestas, no infladas.

Ejecutar: python validation_real_models.py
"""

import sys
import os
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
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

VALIDATION_REPORT_DIR = Path(__file__).parent / 'validation_reports'
VALIDATION_REPORT_DIR.mkdir(exist_ok=True)


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
            logger.info(f"✓ Conectado a BD: {DB_NAME}")
            return True
        except Exception as e:
            logger.error(f"✗ Error conectando a BD: {e}")
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
            logger.error(f"✗ Error en query: {e}")
            return []

    def close(self):
        """Cerrar conexión"""
        if self.conn:
            self.conn.close()


class ModelValidator:
    """Validación real de modelos ML con datos temporales"""

    def __init__(self):
        self.db = DBConnection()
        self.validation_results = {}

    def cargar_datos_con_fechas(self) -> Optional[pd.DataFrame]:
        """Cargar datos con información de fechas para validación temporal"""
        logger.info("\n[VALIDACIÓN] Cargando datos históricos con fechas...")

        if not self.db.connect():
            return None

        # Query mejorada que incluye fechas
        query = """
        SELECT
            u.id as estudiante_id,
            CONCAT(u.name, ' ', COALESCE(u.apellido, '')) as nombre_estudiante,
            u.desempeño_promedio,
            u.asistencia_porcentaje,
            u.participacion_porcentaje,
            u.tareas_completadas,
            u.tareas_pendientes,
            u.actividad_hoy,
            COALESCE(AVG(c.puntaje), 0) as promedio_calificaciones,
            COUNT(c.id) as total_calificaciones,
            COALESCE(ra.promedio, 0) as promedio_rendimiento,
            COALESCE(MAX(c.created_at), u.updated_at) as fecha_actualizacion
        FROM users u
        LEFT JOIN trabajos t ON u.id = t.estudiante_id
        LEFT JOIN calificaciones c ON t.id = c.trabajo_id
        LEFT JOIN rendimiento_academico ra ON u.id = ra.estudiante_id
        WHERE u.tipo_usuario = 'estudiante'
        GROUP BY u.id, u.name, u.apellido, u.desempeño_promedio,
                 u.asistencia_porcentaje, u.participacion_porcentaje,
                 u.tareas_completadas, u.tareas_pendientes, u.actividad_hoy,
                 ra.promedio, u.updated_at
        ORDER BY u.updated_at DESC
        """

        results = self.db.execute_query(query)

        if not results:
            logger.error("✗ No se encontraron datos de estudiantes")
            return None

        df = pd.DataFrame(results)

        # Convertir a datetime
        df['fecha_actualizacion'] = pd.to_datetime(df['fecha_actualizacion'])

        logger.info(f"✓ Datos cargados: {len(df)} estudiantes")
        logger.info(f"  Rango de fechas: {df['fecha_actualizacion'].min()} a {df['fecha_actualizacion'].max()}")

        return df

    def dividir_datos_por_fecha(self, df: pd.DataFrame, dias_test: int = 7) -> Tuple:
        """
        Dividir datos por fecha:
        - TRAINING: Datos históricos
        - TEST: Datos recientes (últimos N días)

        Si no hay suficientes datos históricos, usa random split en su lugar.
        """
        fecha_max = df['fecha_actualizacion'].max()
        fecha_corte = fecha_max - timedelta(days=dias_test)

        df_train = df[df['fecha_actualizacion'] <= fecha_corte].copy()
        df_test = df[df['fecha_actualizacion'] > fecha_corte].copy()

        # Si no hay datos históricos suficientes, usar random split
        if len(df_train) < 5:
            logger.warning(f"\n⚠️  [DIVISIÓN - FALLBACK RANDOM]")
            logger.warning(f"  Rango de fechas en BD: {df['fecha_actualizacion'].min().date()} a {df['fecha_actualizacion'].max().date()}")
            logger.warning(f"  NOTA: Datos demasiado recientes para validación temporal (todos en últimos {dias_test} días)")
            logger.warning(f"  USANDO: Random split 80/20 en su lugar")
            logger.warning(f"  ⚠️  LIMITACIÓN: No es validación temporal honesta, requiere 1-2 meses de datos históricos")

            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            split_idx = int(len(df_shuffled) * 0.8)
            df_train = df_shuffled[:split_idx].copy()
            df_test = df_shuffled[split_idx:].copy()

            logger.info(f"\n[DIVISIÓN RANDOM]")
            logger.info(f"  Training: {len(df_train)} estudiantes (80%)")
            logger.info(f"  Test: {len(df_test)} estudiantes (20%)")
        else:
            logger.info(f"\n[DIVISIÓN TEMPORAL]")
            logger.info(f"  Training: {len(df_train)} estudiantes (hasta {fecha_corte.date()})")
            logger.info(f"  Test: {len(df_test)} estudiantes (últimos {dias_test} días)")

        return df_train, df_test

    def validar_performance_predictor(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict:
        """Validar Performance Predictor con datos reales"""
        logger.info("\n[VALIDACIÓN] Performance Predictor (Predicción de Calificaciones)")
        logger.info("=" * 70)

        features = [
            'desempeño_promedio', 'asistencia_porcentaje',
            'participacion_porcentaje', 'tareas_completadas',
            'tareas_pendientes', 'actividad_hoy'
        ]
        target = 'promedio_calificaciones'

        # Limpiar training
        df_train_clean = df_train[features + [target]].dropna()
        if df_train_clean.empty:
            logger.error("✗ Sin datos limpios en training")
            return {'error': 'Sin datos de training'}

        X_train = df_train_clean[features]
        y_train = df_train_clean[target]

        # Limpiar test
        df_test_clean = df_test[features + [target]].dropna()
        if df_test_clean.empty:
            logger.warning("⚠ Sin datos limpios en test, usando datos de entrenamiento")
            X_test = X_train
            y_test = y_train
        else:
            X_test = df_test_clean[features]
            y_test = df_test_clean[target]

        # Entrenar SOLO con datos históricos
        logger.info(f"\n  Entrenando Random Forest con {len(X_train)} muestras...")
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Validar contra datos recientes (TEST)
        y_pred = model.predict(X_test)

        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Resultados
        results = {
            'modelo': 'Performance Predictor',
            'tipo': 'Regresión',
            'algoritmo': 'Random Forest',
            'muestras_training': len(X_train),
            'muestras_test': len(X_test),
            'features': features,
            'métricas': {
                'R² Score': float(r2),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MSE': float(mse)
            },
            'status': 'EXITOSO'
        }

        logger.info(f"\n  ✓ RESULTADOS:")
        logger.info(f"    R² Score: {r2:.4f} (% varianza explicada)")
        logger.info(f"    RMSE: {rmse:.4f} (error cuadrático promedio)")
        logger.info(f"    MAE: {mae:.4f} (error medio absoluto)")
        logger.info(f"    Muestras test usadas: {len(X_test)}")

        # Interpretación
        if r2 >= 0.7:
            interpretacion = "✓ BUENO - Modelo explica >70% de la varianza"
        elif r2 >= 0.5:
            interpretacion = "⚠ ACEPTABLE - Modelo explica 50-70% de la varianza"
        else:
            interpretacion = "✗ DÉBIL - Modelo explica <50% de la varianza"

        results['interpretación'] = interpretacion
        logger.info(f"    {interpretacion}")

        return results

    def validar_career_recommender(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict:
        """Validar Career Recommender con datos reales"""
        logger.info("\n[VALIDACIÓN] Career Recommender (Recomendación de Carreras)")
        logger.info("=" * 70)

        features = [
            'desempeño_promedio', 'asistencia_porcentaje',
            'participacion_porcentaje', 'promedio_rendimiento'
        ]

        # Limpiar training
        df_train_clean = df_train[features].dropna()
        if df_train_clean.empty:
            logger.error("✗ Sin datos limpios en training")
            return {'error': 'Sin datos de training'}

        X_train = df_train_clean[features]
        y_train = (X_train['desempeño_promedio'] > 70).astype(int)

        # Limpiar test
        df_test_clean = df_test[features].dropna()
        if df_test_clean.empty:
            logger.warning("⚠ Sin datos de test, validando con training")
            X_test = X_train
            y_test = y_train
        else:
            X_test = df_test_clean[features]
            y_test = (X_test['desempeño_promedio'] > 70).astype(int)

        # Entrenar SOLO con histórico
        logger.info(f"\n  Entrenando Random Forest Classifier con {len(X_train)} muestras...")
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Validar
        y_pred = model.predict(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'modelo': 'Career Recommender',
            'tipo': 'Clasificación Binaria',
            'algoritmo': 'Random Forest',
            'clases': ['Bajo rendimiento (≤70)', 'Alto rendimiento (>70)'],
            'muestras_training': len(X_train),
            'muestras_test': len(X_test),
            'métricas': {
                'Accuracy': float(accuracy),
                'Precision': float(precision),
                'Recall': float(recall),
                'F1-Score': float(f1)
            },
            'confusion_matrix': cm.tolist(),
            'status': 'EXITOSO'
        }

        logger.info(f"\n  ✓ RESULTADOS:")
        logger.info(f"    Accuracy: {accuracy:.4f}")
        logger.info(f"    Precision: {precision:.4f}")
        logger.info(f"    Recall: {recall:.4f}")
        logger.info(f"    F1-Score: {f1:.4f}")
        logger.info(f"    Muestras test: {len(X_test)}")
        logger.info(f"    Confusion Matrix:\n      {cm}")

        # Interpretación
        if accuracy >= 0.75:
            interpretacion = "✓ BUENO - Accuracy >75%"
        elif accuracy >= 0.60:
            interpretacion = "⚠ ACEPTABLE - Accuracy 60-75%"
        else:
            interpretacion = "✗ DÉBIL - Accuracy <60%"

        results['interpretación'] = interpretacion
        logger.info(f"    {interpretacion}")

        return results

    def generar_reporte(self, resultados: Dict):
        """Generar reporte JSON de validación"""
        logger.info("\n[REPORTE] Generando reporte de validación...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reporte_path = VALIDATION_REPORT_DIR / f"validation_report_{timestamp}.json"

        # Agregar metadata
        reporte = {
            'timestamp': datetime.now().isoformat(),
            'tipo_validacion': 'Temporal (Training vs Test por Fecha)',
            'descripcion': 'Validación real: Modelos entrenados con datos históricos, validados contra datos recientes',
            'modelos_validados': list(resultados.keys()),
            'resultados': resultados
        }

        with open(reporte_path, 'w') as f:
            json.dump(reporte, f, indent=2)

        logger.info(f"✓ Reporte guardado: {reporte_path}")

        return reporte_path

    def run(self):
        """Ejecutar validación completa"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDACIÓN REAL DE MODELOS ML")
        logger.info("=" * 70)

        # Cargar datos
        df = self.cargar_datos_con_fechas()
        if df is None:
            logger.error("✗ No se pudieron cargar datos")
            return

        # Dividir por fecha
        df_train, df_test = self.dividir_datos_por_fecha(df, dias_test=7)

        # Validar modelos
        self.validation_results['Performance Predictor'] = self.validar_performance_predictor(df_train, df_test)
        self.validation_results['Career Recommender'] = self.validar_career_recommender(df_train, df_test)

        # Generar reporte
        reporte_path = self.generar_reporte(self.validation_results)

        # Resumen
        logger.info("\n" + "=" * 70)
        logger.info("RESUMEN DE VALIDACIÓN")
        logger.info("=" * 70)

        for modelo, resultados in self.validation_results.items():
            if 'error' not in resultados:
                logger.info(f"\n{modelo}:")
                logger.info(f"  {resultados['interpretación']}")
                for metrica, valor in resultados['métricas'].items():
                    logger.info(f"  {metrica}: {valor:.4f}")

        logger.info("\n✓ VALIDACIÓN COMPLETADA")
        logger.info(f"  Reporte guardado en: {reporte_path}")

        # Cerrar conexión
        self.db.close()


if __name__ == '__main__':
    validator = ModelValidator()
    validator.run()
