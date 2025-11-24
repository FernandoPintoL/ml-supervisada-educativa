"""
Módulo de caché para datasets de ML.
Permite guardar y cargar datasets pre-procesados para evitar consultas repetidas a BD.
"""

from .cache_manager import CacheManager, load_cached_dataset, save_dataset

__all__ = ['CacheManager', 'load_cached_dataset', 'save_dataset']
