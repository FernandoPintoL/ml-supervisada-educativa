"""
Script de Entrenamiento: Progress Analyzer (Adaptado)
Plataforma Educativa ML

Entrena el modelo de análisis de progreso usando estructura BD real.

Uso (desde ml_educativas/):
    python -m supervisado.training.train_progress_analyzer
    python -m supervisado.training.train_progress_analyzer --limit 100
    python -m supervisado.training.train_progress_analyzer --save-model

Uso (desde cualquier lado):
    python ml_educativas/supervisado/training/train_progress_analyzer.py
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
from supervisado.models.progress_analyzer import ProgressAnalyzer
from cache.cache_manager import get_cache_manager

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_progress_model(limit: Optional[int] = None,
                         save_model: bool = True,
                         polynomial_degree: int = 2) -> ProgressAnalyzer:
    """
    Entrenar modelo de Progress Analyzer.

    Args:
        limit (int): Límite de estudiantes a cargar
        save_model (bool): Si True, guardar modelo después de entrenar
        polynomial_degree (int): Grado del polinomio (default 2)

    Retorna:
        ProgressAnalyzer: Modelo entrenado
    """
    try:
        logger.info("="*60)
        logger.info("ENTRENAMIENTO: PROGRESS ANALYZER (ADAPTADO)")
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
        logger.info(f"Primeros 3 registros:")
        logger.info(f"\n{data.head(3).to_string()}\n")

        # GUARDAR DATASET EN CACHÉ para acelerar futuras predicciones
        logger.info("\n[2.5/6] Guardando dataset en caché...")
        cache_manager = get_cache_manager()
        metadata = {
            'model': 'ProgressAnalyzer',
            'polynomial_degree': polynomial_degree,
            'limit': limit
        }
        if cache_manager.save_dataset(data, features, metadata):
            logger.info("✓ Dataset guardado en caché exitosamente")
        else:
            logger.warning("⚠ Advertencia: No se pudo guardar dataset en caché")

        # 3. PROCESAR DATOS
        logger.info("[3/6] Procesando datos...")
        processor = DataProcessor(scaler_type="standard")

        # Target: promedio de calificaciones (regresión)
        # El modelo predice la calificación final proyectada
        y_progress = data['promedio_calificaciones'].values

        logger.info(f"Target de progreso:")
        logger.info(f"  Promedio mínimo: {np.min(y_progress):.2f}")
        logger.info(f"  Promedio máximo: {np.max(y_progress):.2f}")
        logger.info(f"  Promedio medio: {np.mean(y_progress):.2f}")
        logger.info(f"  Desviación estándar: {np.std(y_progress):.2f}")

        # Procesar features
        X_processed, _ = processor.process(
            data,
            target_col='promedio_calificaciones',
            features=features,
            fit_scalers=True
        )

        # Alinear y con X_processed (después de remover outliers)
        y_aligned = y_progress[:len(X_processed)] if len(y_progress) >= len(X_processed) else y_progress

        logger.info(f"Datos después de procesar: {len(X_processed)} registros")

        # 4. ENTRENAR MODELO
        logger.info("\n[4/6] Entrenando modelo...")
        model = ProgressAnalyzer(polynomial_degree=polynomial_degree)
        model.set_features(processor.get_feature_names())

        if len(X_processed) < 5:
            logger.error(f"✗ No hay suficientes datos de entrenamiento: {len(X_processed)}")
            return None

        # Convertir a numpy arrays
        X_arr = X_processed.values if hasattr(X_processed, 'values') else X_processed
        y_arr = y_aligned.values if hasattr(y_aligned, 'values') else y_aligned

        # El modelo entrena internamente
        metrics = model.train(X_arr, y_arr)

        logger.info(f"Métricas de entrenamiento:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        # 5. DEMOSTRACIÓN DE PREDICCIONES
        logger.info("\n[5/6] Generando predicciones de ejemplo...")

        # Tomar algunos registros para demostración
        sample_indices = np.random.choice(len(X_arr), min(5, len(X_arr)), replace=False)
        X_sample = X_arr[sample_indices]
        y_sample = y_arr[sample_indices]

        # Obtener predicciones
        progress_predictions = model.predict(X_sample)

        logger.info(f"\nMuestra de análisis de progreso:")
        for i, (pred, actual) in enumerate(zip(progress_predictions, y_sample)):
            error = abs(pred - actual)
            logger.info(f"\n  Estudiante {i}:")
            logger.info(f"    Calificación actual: {actual:.2f}")
            logger.info(f"    Proyección: {pred:.2f}")
            logger.info(f"    Error: {error:.2f} puntos")

            # Learning rate
            learning_rate = ProgressAnalyzer.calculate_learning_rate(
                np.array([actual - (error/2), actual])
            )
            logger.info(f"    Velocidad de aprendizaje: {learning_rate:.2f} puntos/período")

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
        description='Entrenar modelo Progress Analyzer (estructura BD real)'
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
        '--degree',
        type=int,
        default=2,
        help='Grado del polinomio (default 2)'
    )

    args = parser.parse_args()
    save_model = args.save_model and not args.no_save

    # Entrenar modelo
    model = train_progress_model(
        limit=args.limit,
        save_model=save_model,
        polynomial_degree=args.degree
    )

    if model:
        logger.info("\n✓ Modelo listo para usar")
        return 0
    else:
        logger.error("\n✗ Error durante entrenamiento")
        return 1


if __name__ == '__main__':
    exit(main())
