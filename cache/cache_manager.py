"""
CacheManager: Gestiona caché de datasets de ML.
Permite guardar datasets procesados y cargarlos sin acceder a BD.
"""

import os
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """Gestiona caché de datasets de ML"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Inicializa el gestor de caché.

        Args:
            cache_dir: Ruta del directorio de caché.
                      Por defecto: ml_educativas/trained_models/cache/
        """
        if cache_dir is None:
            # Crear en trained_models/cache/ para mantener todo junto
            base_dir = Path(__file__).parent.parent
            cache_dir = base_dir / "trained_models" / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_file = self.cache_dir / "training_data.pkl"
        self.features_file = self.cache_dir / "feature_names.pkl"
        self.metadata_file = self.cache_dir / "metadata.json"

        logger.info(f"CacheManager inicializado en: {self.cache_dir}")

    def save_dataset(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Guardar dataset procesado en caché.

        Args:
            data: DataFrame con datos de entrenamiento
            feature_names: Lista de nombres de features
            metadata: Información adicional (fecha, modelo, versión, etc)

        Returns:
            True si se guardó exitosamente, False en caso contrario
        """
        try:
            # Guardar dataset
            with open(self.dataset_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Dataset guardado: {len(data)} registros, {len(feature_names)} features")

            # Guardar nombres de features
            with open(self.features_file, 'wb') as f:
                pickle.dump(feature_names, f)
            logger.info(f"Feature names guardados: {feature_names}")

            # Guardar metadata
            if metadata is None:
                metadata = {}

            metadata.update({
                'timestamp': datetime.now().isoformat(),
                'num_records': len(data),
                'num_features': len(feature_names),
                'feature_names': feature_names,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
            })

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata guardado: {metadata}")

            return True

        except Exception as e:
            logger.error(f"Error al guardar dataset en caché: {str(e)}")
            return False

    def load_dataset(self) -> Optional[Tuple[pd.DataFrame, List[str]]]:
        """
        Cargar dataset desde caché.

        Returns:
            Tupla (DataFrame, feature_names) si existe, None si no existe
        """
        try:
            if not self.dataset_file.exists():
                logger.warning(f"Cache no encontrado: {self.dataset_file}")
                return None

            # Cargar dataset
            with open(self.dataset_file, 'rb') as f:
                data = pickle.load(f)

            # Cargar feature names
            with open(self.features_file, 'rb') as f:
                feature_names = pickle.load(f)

            logger.info(f"Dataset cargado desde caché: {len(data)} registros")
            return data, feature_names

        except Exception as e:
            logger.error(f"Error al cargar dataset desde caché: {str(e)}")
            return None

    def get_metadata(self) -> Optional[Dict]:
        """
        Obtener metadata del caché.

        Returns:
            Diccionario con metadata, None si no existe
        """
        try:
            if not self.metadata_file.exists():
                return None

            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            return metadata

        except Exception as e:
            logger.error(f"Error al cargar metadata: {str(e)}")
            return None

    def is_cache_valid(self, max_age_hours: int = 24) -> bool:
        """
        Verificar si el caché es válido (existe y no es muy antiguo).

        Args:
            max_age_hours: Máxima antigüedad permitida en horas

        Returns:
            True si caché existe y es reciente, False en caso contrario
        """
        if not self.dataset_file.exists():
            logger.warning("Cache no existe")
            return False

        try:
            metadata = self.get_metadata()
            if not metadata:
                return False

            timestamp_str = metadata.get('timestamp')
            if not timestamp_str:
                return False

            timestamp = datetime.fromisoformat(timestamp_str)
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600

            is_valid = age_hours < max_age_hours
            logger.info(f"Cache antigüedad: {age_hours:.1f} horas - {'Válido' if is_valid else 'Expirado'}")

            return is_valid

        except Exception as e:
            logger.error(f"Error verificando validez de caché: {str(e)}")
            return False

    def clear_cache(self) -> bool:
        """
        Eliminar caché.

        Returns:
            True si se eliminó exitosamente
        """
        try:
            for file in [self.dataset_file, self.features_file, self.metadata_file]:
                if file.exists():
                    file.unlink()
                    logger.info(f"Eliminado: {file}")

            logger.info("Caché limpiado completamente")
            return True

        except Exception as e:
            logger.error(f"Error al limpiar caché: {str(e)}")
            return False

    def get_cache_info(self) -> Dict:
        """
        Obtener información del caché actual.

        Returns:
            Diccionario con información del caché
        """
        metadata = self.get_metadata()

        if not metadata:
            return {
                'exists': False,
                'timestamp': None,
                'num_records': 0,
                'num_features': 0,
                'size_mb': 0
            }

        size_mb = 0
        if self.dataset_file.exists():
            size_mb = self.dataset_file.stat().st_size / (1024 * 1024)

        return {
            'exists': True,
            'timestamp': metadata.get('timestamp'),
            'num_records': metadata.get('num_records', 0),
            'num_features': metadata.get('num_features', 0),
            'size_mb': round(size_mb, 2),
            'feature_names': metadata.get('feature_names', [])
        }


# Singleton global para facilitar uso
_cache_manager = None


def get_cache_manager(cache_dir: Optional[str] = None) -> CacheManager:
    """Obtener instancia global del CacheManager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir)
    return _cache_manager


def load_cached_dataset() -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    Función convenience para cargar dataset desde caché.

    Returns:
        Tupla (DataFrame, feature_names) o None
    """
    manager = get_cache_manager()
    return manager.load_dataset()


def save_dataset(
    data: pd.DataFrame,
    feature_names: List[str],
    metadata: Optional[Dict] = None
) -> bool:
    """
    Función convenience para guardar dataset en caché.

    Args:
        data: DataFrame con datos
        feature_names: Lista de nombres de features
        metadata: Información adicional

    Returns:
        True si se guardó exitosamente
    """
    manager = get_cache_manager()
    return manager.save_dataset(data, feature_names, metadata)
