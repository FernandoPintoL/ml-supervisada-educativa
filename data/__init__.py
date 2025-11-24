"""
Módulo de Datos para Modelos Supervisados
Plataforma Educativa ML

Proporciona herramientas para:
- Cargar datos desde PostgreSQL
- Procesar y limpiar datos
- Ingeniería de features
- Preparación para entrenamiento
"""

from .data_loader import DataLoader
from .data_processor import DataProcessor

__all__ = [
    'DataLoader',
    'DataProcessor',
]

__version__ = '1.0.0'
