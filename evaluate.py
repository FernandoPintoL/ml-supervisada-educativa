"""
Script de Evaluación: Validación Cruzada Avanzada
Plataforma Educativa ML

Evalúa todos los modelos usando K-Fold Cross Validation para obtener
estimaciones realistas de precisión.

Uso (desde ml_educativas/):
    python -m supervisado.evaluate
    python -m supervisado.evaluate --cv 10
    python -m supervisado.evaluate --model performance
    python -m supervisado.evaluate --limit 100

Uso (desde cualquier lado):
    python ml_educativas/supervisado/evaluate.py --limit 50
"""

import sys
import os
import logging
import argparse
from typing import Optional, Dict, List
import json
from datetime import datetime

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
from supervisado.models.career_recommender import CareerRecommender
from supervisado.models.trend_predictor import TrendPredictor
from supervisado.models.progress_analyzer import ProgressAnalyzer

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CVEvaluator:
    """Evaluador de modelos con validación cruzada."""

    def __init__(self, cv_folds: int = 5):
        """
        Inicializar evaluador.

        Args:
            cv_folds (int): Número de folds para K-Fold CV
        """
        self.cv_folds = cv_folds
        self.results = {}

    def evaluate_performance_predictor(self, limit: Optional[int] = None) -> Dict:
        """
        Evaluar Performance Predictor con K-Fold CV.

        Args:
            limit (int): Límite de estudiantes a cargar

        Retorna:
            Dict: Resultados de evaluación
        """
        logger.info("\n" + "="*70)
        logger.info("EVALUANDO: PERFORMANCE PREDICTOR")
        logger.info("="*70)

        try:
            # 1. Cargar datos
            logger.info(f"\n[1/4] Cargando datos...")
            with DataLoaderAdapted() as loader:
                data, features = loader.load_training_data(limit=limit)

            if data.empty:
                logger.error("No hay datos disponibles")
                return {}

            logger.info(f"Datos cargados: {data.shape[0]} estudiantes")

            # 2. Procesar datos
            logger.info(f"\n[2/4] Procesando datos...")
            processor = DataProcessor(scaler_type="standard")

            X_processed, y_raw = processor.process(
                data,
                target_col='promedio_calificaciones',
                features=features,
                fit_scalers=True
            )

            # Target binario
            grade_threshold = y_raw.median()
            y = (y_raw >= grade_threshold).astype(int).values

            logger.info(f"Target: {(y == 1).sum()} sin riesgo, {(y == 0).sum()} en riesgo")

            # 3. Crear modelo
            logger.info(f"\n[3/4] Creando modelo...")
            model = PerformancePredictor()
            model.set_features(processor.get_feature_names())

            # 4. Realizar validación cruzada
            logger.info(f"\n[4/4] Realizando {self.cv_folds}-Fold Cross Validation...")
            X_arr = X_processed.values if hasattr(X_processed, 'values') else X_processed

            cv_results = model.cross_validate_classification(
                X_arr, y,
                cv=self.cv_folds,
                stratified=True
            )

            if cv_results:
                logger.info("\n" + "-"*70)
                logger.info("RESULTADOS DE VALIDACIÓN CRUZADA (PERFORMANCE PREDICTOR)")
                logger.info("-"*70)
                logger.info(f"Accuracy:  {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
                logger.info(f"Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}")
                logger.info(f"Recall:    {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
                logger.info(f"F1-Score:  {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
                logger.info("-"*70)

                self.results['performance_predictor'] = cv_results

            return cv_results

        except Exception as e:
            logger.error(f"Error evaluando Performance Predictor: {str(e)}", exc_info=True)
            return {}

    def evaluate_trend_predictor(self, limit: Optional[int] = None) -> Dict:
        """
        Evaluar Trend Predictor con K-Fold CV.

        Args:
            limit (int): Límite de estudiantes a cargar

        Retorna:
            Dict: Resultados de evaluación
        """
        logger.info("\n" + "="*70)
        logger.info("EVALUANDO: TREND PREDICTOR")
        logger.info("="*70)

        try:
            # 1. Cargar datos
            logger.info(f"\n[1/4] Cargando datos...")
            with DataLoaderAdapted() as loader:
                data, features = loader.load_training_data(limit=limit)

            if data.empty:
                logger.error("No hay datos disponibles")
                return {}

            logger.info(f"Datos cargados: {data.shape[0]} estudiantes")

            # 2. Procesar datos
            logger.info(f"\n[2/4] Procesando datos...")
            processor = DataProcessor(scaler_type="standard")

            X_processed, y_raw = processor.process(
                data,
                target_col='promedio_calificaciones',
                features=features,
                fit_scalers=True
            )

            # Target: tendencia (mejorando/estable/declinando)
            # Simplificado: 0 = declinando, 1 = estable, 2 = mejorando
            y = np.random.randint(0, 3, len(y_raw))  # Simulado para demo

            logger.info(f"Target: {np.bincount(y)}")

            # 3. Crear modelo
            logger.info(f"\n[3/4] Creando modelo...")
            model = TrendPredictor()
            model.set_features(processor.get_feature_names())

            # 4. Realizar validación cruzada
            logger.info(f"\n[4/4] Realizando {self.cv_folds}-Fold Cross Validation...")
            X_arr = X_processed.values if hasattr(X_processed, 'values') else X_processed

            cv_results = model.cross_validate_classification(
                X_arr, y,
                cv=self.cv_folds,
                stratified=True
            )

            if cv_results:
                logger.info("\n" + "-"*70)
                logger.info("RESULTADOS DE VALIDACIÓN CRUZADA (TREND PREDICTOR)")
                logger.info("-"*70)
                logger.info(f"Accuracy:  {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
                logger.info(f"Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}")
                logger.info(f"Recall:    {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
                logger.info(f"F1-Score:  {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
                logger.info("-"*70)

                self.results['trend_predictor'] = cv_results

            return cv_results

        except Exception as e:
            logger.error(f"Error evaluando Trend Predictor: {str(e)}", exc_info=True)
            return {}

    def evaluate_progress_analyzer(self, limit: Optional[int] = None) -> Dict:
        """
        Evaluar Progress Analyzer con K-Fold CV (Regresión).

        Args:
            limit (int): Límite de estudiantes a cargar

        Retorna:
            Dict: Resultados de evaluación
        """
        logger.info("\n" + "="*70)
        logger.info("EVALUANDO: PROGRESS ANALYZER (REGRESIÓN)")
        logger.info("="*70)

        try:
            # 1. Cargar datos
            logger.info(f"\n[1/4] Cargando datos...")
            with DataLoaderAdapted() as loader:
                data, features = loader.load_training_data(limit=limit)

            if data.empty:
                logger.error("No hay datos disponibles")
                return {}

            logger.info(f"Datos cargados: {data.shape[0]} estudiantes")

            # 2. Procesar datos
            logger.info(f"\n[2/4] Procesando datos...")
            processor = DataProcessor(scaler_type="standard")

            X_processed, y = processor.process(
                data,
                target_col='promedio_calificaciones',
                features=features,
                fit_scalers=True
            )

            # 3. Crear modelo
            logger.info(f"\n[3/4] Creando modelo...")
            model = ProgressAnalyzer()
            model.set_features(processor.get_feature_names())

            # 4. Realizar validación cruzada
            logger.info(f"\n[4/4] Realizando {self.cv_folds}-Fold Cross Validation (Regresión)...")
            X_arr = X_processed.values if hasattr(X_processed, 'values') else X_processed
            y_arr = y.values if hasattr(y, 'values') else y

            cv_results = model.cross_validate_regression(
                X_arr, y_arr,
                cv=self.cv_folds
            )

            if cv_results:
                logger.info("\n" + "-"*70)
                logger.info("RESULTADOS DE VALIDACIÓN CRUZADA (PROGRESS ANALYZER)")
                logger.info("-"*70)
                logger.info(f"MSE:  {cv_results['mean_mse']:.4f} ± {cv_results['std_mse']:.4f}")
                logger.info(f"RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
                logger.info(f"MAE:  {cv_results['mean_mae']:.4f} ± {cv_results['std_mae']:.4f}")
                logger.info(f"R²:   {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
                logger.info("-"*70)

                self.results['progress_analyzer'] = cv_results

            return cv_results

        except Exception as e:
            logger.error(f"Error evaluando Progress Analyzer: {str(e)}", exc_info=True)
            return {}

    def print_summary(self):
        """Imprimir resumen de resultados."""
        logger.info("\n" + "="*70)
        logger.info("RESUMEN DE EVALUACIÓN - VALIDACIÓN CRUZADA")
        logger.info("="*70)
        logger.info(f"Folds: {self.cv_folds}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("="*70)

        for model_name, results in self.results.items():
            logger.info(f"\n{model_name.upper().replace('_', ' ')}:")
            if 'mean_accuracy' in results:  # Clasificación
                logger.info(f"  Accuracy:  {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
                logger.info(f"  F1-Score:  {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
            elif 'mean_r2' in results:  # Regresión
                logger.info(f"  R²:        {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
                logger.info(f"  RMSE:      {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")

        logger.info("\n" + "="*70)

    def save_results(self, filepath: str = "cross_validation_results.json"):
        """
        Guardar resultados en JSON.

        Args:
            filepath (str): Ruta del archivo a guardar
        """
        try:
            # Convertir arrays a listas para JSON
            results_serializable = {}
            for model_name, results in self.results.items():
                results_serializable[model_name] = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        results_serializable[model_name][key] = value.tolist()
                    else:
                        results_serializable[model_name][key] = value

            with open(filepath, 'w') as f:
                json.dump(results_serializable, f, indent=2)

            logger.info(f"✓ Resultados guardados en: {filepath}")

        except Exception as e:
            logger.error(f"Error guardando resultados: {str(e)}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Evaluación de modelos ML con Validación Cruzada'
    )
    parser.add_argument('--cv', type=int, default=5, help='Número de folds (default: 5)')
    parser.add_argument('--limit', type=int, default=None, help='Límite de estudiantes')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'performance', 'trend', 'progress'],
                       help='Cuál modelo evaluar')
    parser.add_argument('--save', type=str, default='cross_validation_results.json',
                       help='Archivo para guardar resultados')

    args = parser.parse_args()

    logger.info("\n" + "="*70)
    logger.info("SISTEMA DE EVALUACIÓN CON VALIDACIÓN CRUZADA")
    logger.info("="*70)
    logger.info(f"CV Folds: {args.cv}")
    logger.info(f"Limite estudiantes: {args.limit if args.limit else 'Todos'}")
    logger.info(f"Modelo(s): {args.model}")
    logger.info("="*70)

    # Verificar conexión
    logger.info("\nVerificando conexión a base de datos...")
    if not test_connection():
        logger.error("No se pudo conectar a la base de datos")
        return

    # Crear evaluador
    evaluator = CVEvaluator(cv_folds=args.cv)

    # Evaluar modelos
    if args.model in ['all', 'performance']:
        evaluator.evaluate_performance_predictor(limit=args.limit)

    if args.model in ['all', 'trend']:
        evaluator.evaluate_trend_predictor(limit=args.limit)

    if args.model in ['all', 'progress']:
        evaluator.evaluate_progress_analyzer(limit=args.limit)

    # Imprimir resumen y guardar
    evaluator.print_summary()
    evaluator.save_results(args.save)

    logger.info("\n✓ Evaluación completada")


if __name__ == '__main__':
    main()
