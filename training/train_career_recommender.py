"""
Script de Entrenamiento: Career Recommender (Adaptado)
Plataforma Educativa ML

Entrena el modelo de recomendación de carreras usando estructura BD real.

Uso (desde ml_educativas/):
    python -m supervisado.training.train_career_recommender
    python -m supervisado.training.train_career_recommender --limit 100
    python -m supervisado.training.train_career_recommender --save-model

Uso (desde cualquier lado):
    python ml_educativas/supervisado/training/train_career_recommender.py
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
from supervisado.models.career_recommender import CareerRecommender
from cache.cache_manager import get_cache_manager

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carreras disponibles (8 opciones como se indica en el README)
CAREER_OPTIONS = {
    0: 'Ingeniería en Sistemas',
    1: 'Administración de Empresas',
    2: 'Psicología',
    3: 'Educación',
    4: 'Medicina',
    5: 'Derecho',
    6: 'Contabilidad',
    7: 'Ingeniería Civil'
}


def train_career_model(limit: Optional[int] = None,
                       save_model: bool = True) -> CareerRecommender:
    """
    Entrenar modelo de Career Recommender.

    Args:
        limit (int): Límite de estudiantes a cargar
        save_model (bool): Si True, guardar modelo después de entrenar

    Retorna:
        CareerRecommender: Modelo entrenado
    """
    try:
        logger.info("="*60)
        logger.info("ENTRENAMIENTO: CAREER RECOMMENDER (ADAPTADO)")
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
            'model': 'CareerRecommender',
            'limit': limit
        }
        if cache_manager.save_dataset(data, features, metadata):
            logger.info("✓ Dataset guardado en caché exitosamente")
        else:
            logger.warning("⚠ Advertencia: No se pudo guardar dataset en caché")

        # 3. PROCESAR DATOS
        logger.info("[3/6] Procesando datos...")
        processor = DataProcessor(scaler_type="standard")

        # Target: Asignar carreras aleatoriamente basado en desempeño académico
        # En un sistema real, esto vendría de una columna de BD
        promedio_calificaciones = data['promedio_calificaciones'].values

        # Crear target de carrera basado en desempeño
        # Mejor desempeño -> Carreras más exigentes
        y_career = np.zeros(len(data), dtype=int)
        for i, score in enumerate(promedio_calificaciones):
            if score >= 9.0:
                y_career[i] = 0  # Ingeniería (más exigente)
            elif score >= 8.5:
                y_career[i] = 4  # Medicina (exigente)
            elif score >= 8.0:
                y_career[i] = 5  # Derecho (moderado-alto)
            elif score >= 7.5:
                y_career[i] = 1  # Administración (moderado)
            elif score >= 7.0:
                y_career[i] = 2  # Psicología (moderado)
            elif score >= 6.5:
                y_career[i] = 3  # Educación (moderado)
            elif score >= 6.0:
                y_career[i] = 6  # Contabilidad (moderado-bajo)
            else:
                y_career[i] = 7  # Ingeniería Civil (más flexible)

        logger.info(f"Target de carreras creado:")
        unique_careers, counts = np.unique(y_career, return_counts=True)
        for career_id, count in zip(unique_careers, counts):
            logger.info(f"  {CAREER_OPTIONS[career_id]}: {count} estudiantes")

        # Procesar features
        X_processed, _ = processor.process(
            data,
            target_col='promedio_calificaciones',
            features=features,
            fit_scalers=True
        )

        # Alinear y con X_processed (después de remover outliers)
        y_aligned = y_career[:len(X_processed)] if len(y_career) >= len(X_processed) else y_career

        logger.info(f"Datos después de procesar: {X_processed.shape[0]} registros")

        # Filtrar clases raras (con menos de 2 ejemplos)
        # Necesario para poder hacer stratified split
        unique_classes, class_counts = np.unique(y_aligned, return_counts=True)
        valid_classes = unique_classes[class_counts >= 2]

        valid_mask = np.isin(y_aligned, valid_classes)
        X_processed_filtered = X_processed.iloc[valid_mask] if hasattr(X_processed, 'iloc') else X_processed[valid_mask]
        y_aligned_filtered = y_aligned[valid_mask]

        logger.info(f"Clases raras removidas. Datos finales: {len(X_processed_filtered)} registros")

        # 4. ENTRENAR MODELO
        logger.info("\n[4/6] Entrenando modelo...")
        model = CareerRecommender(career_labels=CAREER_OPTIONS)
        model.set_features(processor.get_feature_names())

        if len(X_processed_filtered) < 5:
            logger.error(f"✗ No hay suficientes datos de entrenamiento: {len(X_processed_filtered)}")
            return None

        # Convertir a numpy arrays
        X_arr = X_processed_filtered.values if hasattr(X_processed_filtered, 'values') else X_processed_filtered
        y_arr = y_aligned_filtered.values if hasattr(y_aligned_filtered, 'values') else y_aligned_filtered

        # El modelo divide internamente en train/val/test
        metrics = model.train(X_arr, y_arr, validation_split=0.2)

        logger.info(f"Métricas de entrenamiento:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # 5. DEMOSTRACIÓN DE PREDICCIONES
        logger.info("\n[5/6] Generando predicciones de ejemplo...")

        # Tomar algunos registros para demostración
        sample_indices = np.random.choice(len(X_arr), min(5, len(X_arr)), replace=False)
        X_sample = X_arr[sample_indices]

        # Obtener predicciones
        career_predictions = model.predict(X_sample)
        career_probs = model.predict_proba(X_sample)

        logger.info(f"\nMuestra de recomendaciones de carrera:")
        for i, pred_idx in enumerate(career_predictions):
            logger.info(f"\n  Estudiante {i}:")
            # Top 3 carreras por probabilidad
            top_indices = np.argsort(career_probs[i])[-3:][::-1]
            for rank, career_idx in enumerate(top_indices, 1):
                confidence = career_probs[i][career_idx]
                career_name = CAREER_OPTIONS.get(career_idx, "Desconocida")
                logger.info(f"    #{rank}: {career_name} ({confidence:.2%})")

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
        description='Entrenar modelo Career Recommender (estructura BD real)'
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
    model = train_career_model(
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
