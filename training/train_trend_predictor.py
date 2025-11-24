"""
Script de Entrenamiento: Trend Predictor (Adaptado)
Plataforma Educativa ML

Entrena el modelo de predicción de tendencias usando estructura BD real.

Uso (desde ml_educativas/):
    python -m supervisado.training.train_trend_predictor
    python -m supervisado.training.train_trend_predictor --limit 100
    python -m supervisado.training.train_trend_predictor --save-model

Uso (desde cualquier lado):
    python ml_educativas/supervisado/training/train_trend_predictor.py
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
from supervisado.models.trend_predictor import TrendPredictor, TREND_LABELS
from cache.cache_manager import get_cache_manager

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def assign_trend_label(grades_sequence: np.ndarray) -> int:
    """
    Asignar etiqueta de tendencia basada en calificaciones.

    Args:
        grades_sequence: Array de calificaciones ordenadas

    Retorna:
        int: Índice de tendencia (0=mejorando, 1=estable, 2=declinando, 3=fluctuante)
    """
    if len(grades_sequence) < 2:
        return 1  # Estable por defecto

    # Calcular features de tendencia
    trend_features = TrendPredictor.calculate_trend_features(grades_sequence)

    slope = trend_features['slope']
    variance = trend_features['variance']
    direction = trend_features['direction']

    # Lógica para clasificar tendencia
    # Basada en pendiente, varianza y dirección

    if variance > 2.0:
        # Alta variabilidad = fluctuante
        return 3
    elif slope > 0.1 and direction > 0.5:
        # Pendiente positiva = mejorando
        return 0
    elif slope < -0.1 and direction < -0.5:
        # Pendiente negativa = declinando
        return 2
    else:
        # Pendiente cercana a cero = estable
        return 1


def train_trend_model(limit: Optional[int] = None,
                      save_model: bool = True) -> TrendPredictor:
    """
    Entrenar modelo de Trend Predictor.

    Args:
        limit (int): Límite de estudiantes a cargar
        save_model (bool): Si True, guardar modelo después de entrenar

    Retorna:
        TrendPredictor: Modelo entrenado
    """
    try:
        logger.info("="*60)
        logger.info("ENTRENAMIENTO: TREND PREDICTOR (ADAPTADO)")
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
            'model': 'TrendPredictor',
            'limit': limit
        }
        if cache_manager.save_dataset(data, features, metadata):
            logger.info("✓ Dataset guardado en caché exitosamente")
        else:
            logger.warning("⚠ Advertencia: No se pudo guardar dataset en caché")

        # 3. PROCESAR DATOS
        logger.info("[3/6] Procesando datos...")
        processor = DataProcessor(scaler_type="standard")

        # Crear target de tendencia basado en varianza y pendiente
        # En un sistema real, esto vendría de análisis temporal de calificaciones
        y_trend = np.zeros(len(data), dtype=int)

        for i, row in data.iterrows():
            # Usar varianza como proxy para tendencia
            varianza = row['varianza_calificaciones']
            promedio = row['promedio_calificaciones']
            num_califs = row['num_calificaciones']

            if varianza > 3.5:
                # Alta varianza = fluctuante
                y_trend[i] = 3
            elif promedio >= 8.0:
                # Alto promedio = mejorando
                y_trend[i] = 0
            elif promedio < 6.0 and num_califs >= 10:
                # Bajo promedio con muchas calificaciones = declinando
                y_trend[i] = 2
            else:
                # Promedio medio = estable
                y_trend[i] = 1

        logger.info(f"Target de tendencia creado:")
        unique_trends, counts = np.unique(y_trend, return_counts=True)
        for trend_id, count in zip(unique_trends, counts):
            trend_name = TREND_LABELS.get(trend_id, "Desconocida")
            logger.info(f"  {trend_name}: {count} estudiantes")

        # Procesar features
        X_processed, _ = processor.process(
            data,
            target_col='promedio_calificaciones',
            features=features,
            fit_scalers=True
        )

        # Alinear y con X_processed (después de remover outliers)
        y_aligned = y_trend[:len(X_processed)] if len(y_trend) >= len(X_processed) else y_trend

        logger.info(f"Datos después de procesar: {len(X_processed)} registros")

        # 4. ENTRENAR MODELO
        logger.info("\n[4/6] Entrenando modelo...")
        model = TrendPredictor()
        model.set_features(processor.get_feature_names())

        if len(X_processed) < 5:
            logger.error(f"✗ No hay suficientes datos de entrenamiento: {len(X_processed)}")
            return None

        # Convertir a numpy arrays
        X_arr = X_processed.values if hasattr(X_processed, 'values') else X_processed
        y_arr = y_aligned.values if hasattr(y_aligned, 'values') else y_aligned

        # El modelo divide internamente en train/val/test
        metrics = model.train(X_arr, y_arr)

        logger.info(f"Métricas de entrenamiento:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # 5. DEMOSTRACIÓN DE PREDICCIONES
        logger.info("\n[5/6] Generando predicciones de ejemplo...")

        # Tomar algunos registros para demostración
        sample_indices = np.random.choice(len(X_arr), min(5, len(X_arr)), replace=False)
        X_sample = X_arr[sample_indices]

        # Obtener predicciones
        trend_predictions = model.predict(X_sample)
        trend_probs = model.predict_proba(X_sample)

        logger.info(f"\nMuestra de predicciones de tendencia:")
        for i, pred_idx in enumerate(trend_predictions):
            logger.info(f"\n  Estudiante {i}:")
            pred_name = TREND_LABELS.get(pred_idx, "Desconocida")
            confidence = np.max(trend_probs[i]) if hasattr(trend_probs, '__getitem__') else 0
            logger.info(f"    Predicción: {pred_name}")
            logger.info(f"    Confianza: {confidence:.2%}")

            # Top 3 tendencias
            if hasattr(trend_probs, '__getitem__'):
                top_indices = np.argsort(trend_probs[i])[-3:][::-1]
                logger.info(f"    Top tendencias:")
                for rank, trend_idx in enumerate(top_indices, 1):
                    trend_name = TREND_LABELS.get(trend_idx, "Desconocida")
                    prob = trend_probs[i][trend_idx]
                    logger.info(f"      #{rank}: {trend_name} ({prob:.2%})")

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
        description='Entrenar modelo Trend Predictor (estructura BD real)'
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

    args = parser.parse_args()
    save_model = args.save_model and not args.no_save

    # Entrenar modelo
    model = train_trend_model(
        limit=args.limit,
        save_model=save_model
    )

    if model:
        logger.info("\n✓ Modelo listo para usar")
        return 0
    else:
        logger.error("\n✗ Error durante entrenamiento")
        return 1


if __name__ == '__main__':
    exit(main())
