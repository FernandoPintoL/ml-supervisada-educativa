"""
Script de Explicabilidad: SHAP Explanations
Plataforma Educativa ML

Genera explicaciones SHAP para predicciones de modelos.

Uso (desde ml_educativas/):
    python -m supervisado.explain_predictions
    python -m supervisado.explain_predictions --model performance --limit 10
    python -m supervisado.explain_predictions --student-id 5

Uso (desde cualquier lado):
    python ml_educativas/supervisado/explain_predictions.py --limit 20
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
supervisado_dir = os.path.dirname(current_file)
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


def explain_performance_predictor(limit: Optional[int] = None,
                                  num_explanations: int = 5) -> None:
    """
    Generar explicaciones SHAP para Performance Predictor.

    Args:
        limit (int): Límite de estudiantes a cargar
        num_explanations (int): Número de predicciones a explicar
    """
    try:
        logger.info("="*70)
        logger.info("EXPLICABILIDAD: PERFORMANCE PREDICTOR")
        logger.info("="*70)

        # 1. Verificar conexión
        logger.info("\n[1/4] Verificando conexión a base de datos...")
        if not test_connection():
            logger.error("✗ No se pudo conectar a la base de datos")
            return

        # 2. Cargar datos
        logger.info("\n[2/4] Cargando datos...")
        with DataLoaderAdapted() as loader:
            data, features = loader.load_training_data(limit=limit)

        if data.empty:
            logger.error("✗ No hay datos disponibles")
            return

        logger.info(f"Datos cargados: {data.shape}")

        # 3. Procesar datos
        logger.info("\n[3/4] Procesando datos...")
        processor = DataProcessor(scaler_type="standard")

        X_processed, y_raw = processor.process(
            data,
            target_col='promedio_calificaciones',
            features=features,
            fit_scalers=True
        )

        # Target binario
        median_grade = y_raw.median()
        y = (y_raw >= median_grade).astype(int).values

        # Entrenar modelo
        model = PerformancePredictor()
        model.set_features(processor.get_feature_names())

        X_train_arr = X_processed.values if hasattr(X_processed, 'values') else X_processed
        y_train_arr = y

        metrics = model.train(X_train_arr, y_train_arr)
        logger.info(f"Modelo entrenado: accuracy={metrics.get('train_score', 0):.4f}")

        # 4. Generar explicaciones
        logger.info(f"\n[4/4] Generando {num_explanations} explicaciones SHAP...")

        for i in range(min(num_explanations, len(X_train_arr))):
            logger.info(f"\n{'='*70}")
            logger.info(f"PREDICCIÓN {i+1}/{num_explanations}")
            logger.info(f"{'='*70}")

            # Explicar predicción individual
            explanation = model.explain_prediction(
                X_train_arr,
                sample_index=i,
                feature_names=processor.get_feature_names(),
                max_display=5
            )

            if explanation:
                logger.info(f"\n{explanation['explanation_text']}")

                logger.info(f"\nContribuciones de features:")
                for contrib in explanation['feature_contributions']:
                    logger.info(
                        f"  • {contrib['feature']}: {contrib['contribution']:.6f} "
                        f"({contrib['impact']})"
                    )

        # Importancia global de features
        logger.info(f"\n{'='*70}")
        logger.info("IMPORTANCIA GLOBAL DE FEATURES (SHAP)")
        logger.info(f"{'='*70}")

        importance = model.get_feature_importance_shap(
            X_train_arr,
            feature_names=processor.get_feature_names()
        )

        if importance:
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_importance:
                bar_length = int(score / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                logger.info(f"  {feature:30s} {bar} {score:.2f}%")

        logger.info("\n" + "="*70)
        logger.info("✓ EXPLICACIONES GENERADAS EXITOSAMENTE")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"✗ Error generando explicaciones: {str(e)}", exc_info=True)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Generación de Explicaciones SHAP para Predicciones'
    )
    parser.add_argument('--model', type=str, default='performance',
                       choices=['performance', 'trend', 'all'],
                       help='Cuál modelo explicar')
    parser.add_argument('--limit', type=int, default=None,
                       help='Límite de estudiantes a cargar')
    parser.add_argument('--num-explanations', type=int, default=5,
                       help='Número de explicaciones a generar')
    parser.add_argument('--student-id', type=int, default=None,
                       help='ID específico de estudiante a explicar')

    args = parser.parse_args()

    if args.model in ['performance', 'all']:
        explain_performance_predictor(
            limit=args.limit,
            num_explanations=args.num_explanations
        )


if __name__ == '__main__':
    main()
