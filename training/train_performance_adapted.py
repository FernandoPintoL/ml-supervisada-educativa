"""
Script de Entrenamiento: Performance Predictor (Adaptado)
Plataforma Educativa ML

Entrena el modelo de predicción de riesgo académico usando estructura BD real.

Uso (desde ml_educativas/):
    python -m supervisado.training.train_performance_adapted
    python -m supervisado.training.train_performance_adapted --limit 100
    python -m supervisado.training.train_performance_adapted --save-model

Uso (desde cualquier lado):
    python ml_educativas/supervisado/training/train_performance_adapted.py
"""

import sys
import os
import logging
import argparse
from typing import Optional

import numpy as np
import pandas as pd

# Agregar ml_educativas al path
current_file = os.path.abspath(__file__)
supervisado_dir = os.path.dirname(os.path.dirname(current_file))
ml_educativas_dir = os.path.dirname(supervisado_dir)

if ml_educativas_dir not in sys.path:
    sys.path.insert(0, ml_educativas_dir)

from shared.database.connection import test_connection
from shared.config import DEBUG, LOG_LEVEL, MODELS_DIR
from supervisado.data.data_loader_adapted import DataLoaderAdapted
from supervisado.data.data_processor import DataProcessor
from supervisado.models.performance_predictor import PerformancePredictor

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_performance_model(limit: Optional[int] = None,
                           save_model: bool = True,
                           grade_threshold: float = 5.0) -> PerformancePredictor:
    """
    Entrenar modelo de Performance Predictor.

    Args:
        limit (int): Límite de estudiantes a cargar
        save_model (bool): Si True, guardar modelo después de entrenar
        grade_threshold (float): Calificación mínima para considerar sin riesgo

    Retorna:
        PerformancePredictor: Modelo entrenado
    """
    try:
        logger.info("="*60)
        logger.info("ENTRENAMIENTO: PERFORMANCE PREDICTOR (ADAPTADO)")
        logger.info("="*60)

        # 1. VERIFICAR CONEXIÓN
        logger.info("\n[1/6] Verificando conexión a base de datos...")
        if not test_connection():
            logger.error("✗ No se pudo conectar a la base de datos")
            return None

        # 2. CARGAR DATOS
        logger.info("\n[2/6] Cargando datos de BD real...")
        with DataLoaderAdapted() as loader:
            data, features = loader.load_training_data(limit=limit)

        if data.empty:
            logger.error("✗ No hay datos disponibles para entrenamiento")
            return None

        logger.info(f"Datos cargados: {data.shape}")
        logger.info(f"Features disponibles: {features}")

        # Mostrar primeros registros
        logger.info(f"\nPrimeros 3 registros:")
        logger.info(data.head(3).to_string())

        # 3. PROCESAR DATOS
        logger.info("\n[3/6] Procesando datos...")
        processor = DataProcessor(scaler_type="standard")

        # Target: promedio_calificaciones
        if 'promedio_calificaciones' not in data.columns:
            logger.error("✗ No se encontró columna 'promedio_calificaciones'")
            logger.info(f"Columnas disponibles: {data.columns.tolist()}")
            return None

        # Procesar
        X_processed, y_raw = processor.process(
            data,
            target_col='promedio_calificaciones',
            features=features,
            fit_scalers=True
        )

        # Crear variable binaria de riesgo
        if y_raw is None:
            logger.error("No se pudo extraer target")
            return None

        # IMPORTANTE: Alinear y con X_processed (después de remover outliers)
        # X_processed tiene índices válidos después de remover outliers
        y_raw_aligned = y_raw.loc[X_processed.index] if hasattr(X_processed, 'index') else y_raw[:len(X_processed)]

        y = (y_raw_aligned >= grade_threshold).astype(int).values
        num_risk = (y == 0).sum()
        num_ok = (y == 1).sum()

        logger.info(f"Target binario creado:")
        logger.info(f"  Sin riesgo (≥ {grade_threshold}): {num_ok} estudiantes")
        logger.info(f"  En riesgo (< {grade_threshold}): {num_risk} estudiantes")

        if num_risk == 0 or num_ok == 0:
            logger.warning("⚠ Una de las clases no tiene ejemplos. Usando threshold diferente.")
            median_grade = y_raw_aligned.median()
            logger.info(f"Usando mediana como threshold: {median_grade:.2f}")
            y = (y_raw_aligned >= median_grade).astype(int).values
            num_risk = (y == 0).sum()
            num_ok = (y == 1).sum()
            logger.info(f"  Clase 0: {num_risk}, Clase 1: {num_ok}")

        # Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = processor.train_val_test_split(
            X_processed, y
        )

        logger.info(f"Splits: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

        # 4. ENTRENAR MODELO
        logger.info("\n[4/6] Entrenando modelo...")
        model = PerformancePredictor()
        model.set_features(processor.get_feature_names())

        if X_train.shape[0] < 5:
            logger.error(f"✗ No hay suficientes datos de entrenamiento: {X_train.shape[0]}")
            return None

        # X_train and y_train might already be numpy arrays
        X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train

        metrics = model.train(X_train_arr, y_train_arr)

        logger.info(f"Métricas de entrenamiento:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # 5. EVALUAR EN CONJUNTO DE PRUEBA
        logger.info("\n[5/6] Evaluando en conjunto de prueba...")
        if X_test.shape[0] > 0:
            X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
            y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test

            y_test_pred = model.predict(X_test_arr)
            test_metrics = model.evaluate_classification(y_test_arr, y_test_pred)

            logger.info(f"Métricas de prueba:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            # Predicciones de riesgo
            y_test_proba = model.predict_proba(X_test_arr).flatten()
            risk_levels = model.predict_risk_level(X_test_arr)

            logger.info(f"\nMuestra de predicciones de riesgo:")
            for i in range(min(5, len(risk_levels))):
                logger.info(
                    f"  Estudiante {i}: "
                    f"riesgo={risk_levels[i]['risk_level']}, "
                    f"score={risk_levels[i]['risk_score']:.2f}"
                )

        # 6. GUARDAR MODELO
        logger.info("\n[6/6] Guardando modelo...")
        if save_model:
            filepath = model.save(directory=MODELS_DIR)
            logger.info(f"✓ Modelo guardado en: {filepath}")
        else:
            logger.info("Modelo no guardado (--save-model no activado)")

        logger.info("\n" + "="*60)
        logger.info("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*60)

        return model

    except Exception as e:
        logger.error(f"✗ Error durante entrenamiento: {str(e)}", exc_info=True)
        return None


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo Performance Predictor (estructura BD real)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Límite de estudiantes a cargar'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=True,
        help='Guardar modelo después de entrenar'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='No guardar modelo'
    )
    parser.add_argument(
        '--grade-threshold',
        type=float,
        default=5.0,
        help='Calificación mínima para considerar sin riesgo (default: 5.0)'
    )

    args = parser.parse_args()
    save_model = args.save_model and not args.no_save

    # Entrenar modelo
    model = train_performance_model(
        limit=args.limit,
        save_model=save_model,
        grade_threshold=args.grade_threshold
    )

    if model:
        logger.info("\n✓ Modelo listo para usar")
        return 0
    else:
        logger.error("\n✗ Error durante entrenamiento")
        return 1


if __name__ == '__main__':
    exit(main())
