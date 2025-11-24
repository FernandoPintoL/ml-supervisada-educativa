"""
Script de Entrenamiento: Performance Predictor
Plataforma Educativa ML

Entrena el modelo de predicción de riesgo académico.

Uso (desde ml_educativas/):
    python -m supervisado.training.train_performance
    python -m supervisado.training.train_performance --limit 1000
    python -m supervisado.training.train_performance --save-model

Uso (desde cualquier lado):
    python ml_educativas/supervisado/training/train_performance.py
"""

import sys
import os
import logging
import argparse
from typing import Optional

import numpy as np
import pandas as pd

# Agregar ml_educativas al path
# El script está en: ml_educativas/supervisado/training/train_performance.py
# Necesitamos llegar a: ml_educativas/
current_file = os.path.abspath(__file__)
supervisado_dir = os.path.dirname(os.path.dirname(current_file))  # ml_educativas/supervisado
ml_educativas_dir = os.path.dirname(supervisado_dir)  # ml_educativas/

# Agregar ml_educativas al path para importar sus módulos
sys.path.insert(0, ml_educativas_dir)

# Ahora importar directamente (sin ml_educativas. prefix)
from shared.database.connection import DBSession, test_connection
from shared.config import DEBUG, LOG_LEVEL, MODELS_DIR
from supervisado.data.data_loader import DataLoader
from supervisado.data.data_processor import DataProcessor
from supervisado.models.performance_predictor import PerformancePredictor
from cache.cache_manager import get_cache_manager

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_performance_model(limit: Optional[int] = None,
                           save_model: bool = True,
                           grade_threshold: float = 3.0) -> PerformancePredictor:
    """
    Entrenar modelo de Performance Predictor.

    Args:
        limit (int): Límite de estudiantes a cargar
        save_model (bool): Si True, guardar modelo después de entrenar
        grade_threshold (float): Nota mínima para considerar sin riesgo

    Retorna:
        PerformancePredictor: Modelo entrenado
    """
    try:
        logger.info("="*60)
        logger.info("ENTRENAMIENTO: PERFORMANCE PREDICTOR")
        logger.info("="*60)

        # 1. VERIFICAR CONEXIÓN
        logger.info("\n[1/6] Verificando conexión a base de datos...")
        if not test_connection():
            logger.error("✗ No se pudo conectar a la base de datos")
            return None

        # 2. CARGAR DATOS
        logger.info("\n[2/6] Cargando datos...")
        with DataLoader() as loader:
            data, features = loader.load_training_data(limit=limit)

        if data.empty:
            logger.error("✗ No hay datos disponibles para entrenamiento")
            return None

        logger.info(f"Datos cargados: {data.shape}")
        logger.info(f"Features disponibles: {features}")

        # GUARDAR DATASET EN CACHÉ para acelerar futuras predicciones
        logger.info("\n[2.5/6] Guardando dataset en caché...")
        cache_manager = get_cache_manager()
        metadata = {
            'model': 'PerformancePredictor',
            'grade_threshold': grade_threshold,
            'limit': limit
        }
        if cache_manager.save_dataset(data, features, metadata):
            logger.info("✓ Dataset guardado en caché exitosamente")
        else:
            logger.warning("⚠ Advertencia: No se pudo guardar dataset en caché")

        # 3. PROCESAR DATOS
        logger.info("\n[3/6] Procesando datos...")
        processor = DataProcessor(scaler_type="standard")

        # Procesar datos
        X_processed, y_raw = processor.process(
            data,
            target_col='promedio_ultimas_notas',
            features=features,
            fit_scalers=True
        )

        # Crear variable binaria de riesgo (alineada con X_processed)
        if y_raw is None:
            logger.error("No se pudo extraer target")
            return None

        # Asegurar que y tiene la misma longitud que X_processed
        y = (y_raw >= grade_threshold).astype(int).values

        # Si X_processed y y tienen diferentes longitudes, sincronizar
        if len(X_processed) != len(y):
            logger.warning(f"Desalineación detectada: X_processed={len(X_processed)}, y={len(y)}")
            # Usar los índices de X_processed para sincronizar y
            if hasattr(X_processed, 'index'):
                y = y[X_processed.index]
            else:
                # Si X_processed es numpy, tomar los primeros N elementos
                y = y[:len(X_processed)]

        num_risk = (y == 0).sum()
        num_ok = (y == 1).sum()

        logger.info(f"Target binario creado:")
        logger.info(f"  Sin riesgo: {num_ok} estudiantes")
        logger.info(f"  En riesgo: {num_risk} estudiantes")

        # Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = processor.train_val_test_split(
            X_processed, y
        )

        # 4. ENTRENAR MODELO
        logger.info("\n[4/6] Entrenando modelo...")
        model = PerformancePredictor()
        model.set_features(processor.get_feature_names())

        # Convertir a numpy arrays si son DataFrames
        X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_values = y_train.values if hasattr(y_train, 'values') else y_train

        metrics = model.train(X_train_values, y_train_values)

        logger.info(f"Métricas de entrenamiento:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # 5. EVALUAR EN CONJUNTO DE PRUEBA
        logger.info("\n[5/6] Evaluando en conjunto de prueba...")
        X_test_values = X_test.values if hasattr(X_test, 'values') else X_test
        y_test_values = y_test.values if hasattr(y_test, 'values') else y_test

        y_test_pred = model.predict(X_test_values)
        test_metrics = model.evaluate_classification(y_test_values, y_test_pred)

        logger.info(f"Métricas de prueba:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Predicciones de riesgo
        y_test_proba = model.predict_proba(X_test_values).flatten()
        risk_levels = model.predict_risk_level(X_test_values)

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
        description='Entrenar modelo Performance Predictor'
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
        default=3.0,
        help='Nota mínima para considerar sin riesgo'
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
