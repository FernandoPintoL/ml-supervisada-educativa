"""
Modelos de Aprendizaje Supervisado
Plataforma Educativa ML

Proporciona acceso a todos los modelos supervisados:
- PerformancePredictor: Predicción de riesgo académico (Random Forest + XGBoost)
- CareerRecommender: Recomendación de carreras (SVM + KNN)
- TrendPredictor: Predicción de tendencias (XGBoost multiclase)
- ProgressAnalyzer: Análisis de progreso (Regresión Lineal + Polinomial)
"""

from .base_model import BaseModel
from .performance_predictor import PerformancePredictor
from .career_recommender import CareerRecommender
from .trend_predictor import TrendPredictor
from .progress_analyzer import ProgressAnalyzer

__all__ = [
    'BaseModel',
    'PerformancePredictor',
    'CareerRecommender',
    'TrendPredictor',
    'ProgressAnalyzer',
]

__version__ = '1.0.0'
