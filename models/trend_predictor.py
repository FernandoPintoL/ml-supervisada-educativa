"""
Trend Predictor Model
Plataforma Educativa ML

Predice tendencias en el rendimiento académico usando:
- XGBoost Multiclase para clasificación de tendencias
- Análisis de series temporales implícitas

Categorías de tendencia:
- Mejorando: Calificaciones ascendentes
- Estable: Calificaciones consistentes
- Declinando: Calificaciones descendentes
- Fluctuante: Calificaciones variables
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score

from .base_model import BaseModel
from shared.config import (
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_RANDOM_STATE,
    TEST_SIZE,
    DEBUG
)

# Configurar logger
logger = logging.getLogger(__name__)

# Mapeo de tendencias
TREND_LABELS = {
    0: 'improving',      # Mejorando
    1: 'stable',         # Estable
    2: 'declining',      # Declinando
    3: 'fluctuating'     # Fluctuante
}

TREND_NAMES = {v: k for k, v in TREND_LABELS.items()}


class TrendPredictor(BaseModel):
    """
    Modelo para predecir tendencias de rendimiento académico.

    Utiliza XGBoost multiclase para clasificar tipos de tendencia.

    Atributos:
        xgb_model (XGBClassifier): Modelo XGBoost multiclase
        trend_labels (Dict[int, str]): Mapeo de índices a nombres de tendencia
        cross_val_scores: Scores de validación cruzada
    """

    def __init__(self):
        """Inicializar Trend Predictor."""
        super().__init__(name="TrendPredictor", model_type="supervisado")

        # Modelo XGBoost para multiclase
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=XGB_RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss',  # Para multiclase
            verbosity=0,
            objective='multi:softprob'  # Multiclase
        )

        self.trend_labels = TREND_LABELS.copy()
        self.cross_val_scores = None
        self.class_distribution = {}

        logger.info(f"✓ {self.name} inicializado")

    # ===========================================
    # EXTRACCIÓN DE CARACTERÍSTICAS DE TENDENCIA
    # ===========================================

    @staticmethod
    def calculate_trend_features(grades_sequence: np.ndarray) -> Dict[str, float]:
        """
        Calcular features de tendencia desde una secuencia de calificaciones.

        Features:
        - slope: Pendiente de la línea de tendencia (positiva = mejora)
        - variancia: Variabilidad de calificaciones
        - direction: Cambio en últimas notas
        - momentum: Aceleración del cambio

        Args:
            grades_sequence (np.ndarray): Secuencia de calificaciones ordenadas temporalmente

        Retorna:
            Dict[str, float]: Features de tendencia
        """
        try:
            if len(grades_sequence) < 2:
                return {
                    'slope': 0.0,
                    'variance': 0.0,
                    'direction': 0.0,
                    'momentum': 0.0,
                    'mean_grade': float(grades_sequence[0]) if len(grades_sequence) > 0 else 0.0
                }

            # 1. Slope: regresión lineal simple
            x = np.arange(len(grades_sequence))
            slope = np.polyfit(x, grades_sequence, 1)[0]

            # 2. Variancia: consistencia
            variance = np.var(grades_sequence)

            # 3. Direction: cambio en últimas 3 notas
            if len(grades_sequence) >= 3:
                direction = grades_sequence[-1] - grades_sequence[-3]
            else:
                direction = grades_sequence[-1] - grades_sequence[0]

            # 4. Momentum: aceleración (segunda derivada)
            if len(grades_sequence) >= 3:
                diff1 = np.diff(grades_sequence)
                momentum = diff1[-1] - diff1[0]
            else:
                momentum = 0.0

            # 5. Media
            mean_grade = np.mean(grades_sequence)

            return {
                'slope': float(slope),
                'variance': float(variance),
                'direction': float(direction),
                'momentum': float(momentum),
                'mean_grade': float(mean_grade)
            }

        except Exception as e:
            logger.error(f"Error calculando features de tendencia: {str(e)}")
            return {
                'slope': 0.0,
                'variance': 0.0,
                'direction': 0.0,
                'momentum': 0.0,
                'mean_grade': 0.0
            }

    @staticmethod
    def classify_trend(slope: float, variance: float, direction: float) -> int:
        """
        Clasificar tendencia basada en slope, variance y direction.

        Reglas:
        - Mejorando: slope > 0.1 AND direction > 0
        - Declinando: slope < -0.1 AND direction < 0
        - Fluctuante: variance > 1.5
        - Estable: resto

        Args:
            slope (float): Pendiente de la línea de tendencia
            variance (float): Variancia de calificaciones
            direction (float): Cambio reciente

        Retorna:
            int: Índice de tendencia (0-3)
        """
        # Thresholds
        SLOPE_THRESHOLD = 0.15
        VARIANCE_THRESHOLD = 1.5
        DIRECTION_THRESHOLD = 0.3

        if variance > VARIANCE_THRESHOLD:
            return TREND_NAMES['fluctuating']  # Fluctuante

        if slope > SLOPE_THRESHOLD and direction > DIRECTION_THRESHOLD:
            return TREND_NAMES['improving']  # Mejorando

        if slope < -SLOPE_THRESHOLD and direction < -DIRECTION_THRESHOLD:
            return TREND_NAMES['declining']  # Declinando

        return TREND_NAMES['stable']  # Estable

    # ===========================================
    # PREPARACIÓN DE DATOS
    # ===========================================

    def _create_trend_labels(self, df: pd.DataFrame,
                            grades_by_student: Dict[int, list]) -> np.ndarray:
        """
        Crear labels de tendencia a partir de secuencias de calificaciones.

        Args:
            df (DataFrame): Datos de estudiantes
            grades_by_student (Dict[int, list]): Calificaciones por estudiante

        Retorna:
            np.ndarray: Array de labels de tendencia
        """
        try:
            trends = []

            for _, row in df.iterrows():
                student_id = row['id'] if 'id' in row else row['student_id']

                if student_id in grades_by_student:
                    grades_seq = np.array(grades_by_student[student_id])

                    # Calcular features
                    features = self.calculate_trend_features(grades_seq)

                    # Clasificar tendencia
                    trend_idx = self.classify_trend(
                        features['slope'],
                        features['variance'],
                        features['direction']
                    )
                    trends.append(trend_idx)
                else:
                    trends.append(TREND_NAMES['stable'])  # Default

            return np.array(trends)

        except Exception as e:
            logger.error(f"Error creando labels de tendencia: {str(e)}")
            return np.zeros(len(df), dtype=int)

    def _analyze_class_distribution(self, y: np.ndarray) -> None:
        """
        Analizar distribución de clases de tendencia.

        Args:
            y (np.ndarray): Variable target
        """
        try:
            unique, counts = np.unique(y, return_counts=True)
            self.class_distribution = dict(zip(unique, counts))

            logger.info("Distribución de tendencias:")
            for trend_idx, count in sorted(self.class_distribution.items()):
                trend_name = self.trend_labels.get(trend_idx, f"Tendencia {trend_idx}")
                pct = (count / len(y)) * 100
                logger.info(f"  {trend_name}: {count} estudiantes ({pct:.1f}%)")

        except Exception as e:
            logger.error(f"Error analizando distribución: {str(e)}")

    # ===========================================
    # ENTRENAMIENTO
    # ===========================================

    def train(self, X: np.ndarray, y: np.ndarray,
             validation_split: float = 0.2,
             **kwargs) -> Dict[str, float]:
        """
        Entrenar el modelo de tendencias (XGBoost multiclase).

        Args:
            X (np.ndarray): Features (n_samples, n_features)
            y (np.ndarray): Target multiclase - Índices de tendencia (0-3)
            validation_split (float): Proporción para validación
            **kwargs: Argumentos adicionales

        Retorna:
            Dict[str, float]: Métricas de entrenamiento
        """
        try:
            logger.info("Iniciando entrenamiento de TrendPredictor...")

            # Analizar distribución
            self._analyze_class_distribution(y)

            # Dividir datos
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )

            # ========== ENTRENAR XGBOOST ==========
            logger.info("Entrenando XGBoost multiclase...")
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            train_score = self.xgb_model.score(X_train, y_train)
            val_score = self.xgb_model.score(X_val, y_val)

            logger.info(
                f"XGBoost: train_score={train_score:.4f}, "
                f"val_score={val_score:.4f}"
            )

            # ========== FEATURE IMPORTANCE ==========
            self.feature_importance = dict(zip(
                range(X.shape[1]),
                self.xgb_model.feature_importances_
            ))

            # ========== CROSS VALIDATION ==========
            logger.info("Calculando cross-validation...")
            self.cross_val_scores = cross_val_score(
                xgb.XGBClassifier(
                    n_estimators=XGB_N_ESTIMATORS,
                    max_depth=XGB_MAX_DEPTH,
                    learning_rate=XGB_LEARNING_RATE,
                    random_state=XGB_RANDOM_STATE,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                ),
                X, y, cv=5, n_jobs=-1
            )

            logger.info(
                f"Cross-validation: {self.cross_val_scores.mean():.4f} "
                f"(± {self.cross_val_scores.std():.4f})"
            )

            # Marcar como entrenado
            self.is_trained = True
            self.features = list(range(X.shape[1]))

            # Retornar métricas
            metrics = {
                'train_score': float(train_score),
                'val_score': float(val_score),
                'cv_mean': float(self.cross_val_scores.mean()),
                'cv_std': float(self.cross_val_scores.std()),
                'num_trends': len(np.unique(y))
            }

            self.metadata['metrics'] = metrics
            logger.info(f"✓ Entrenamiento completado")

            return metrics

        except Exception as e:
            logger.error(f"✗ Error entrenando modelo: {str(e)}")
            raise

    # ===========================================
    # PREDICCIÓN
    # ===========================================

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predecir tendencia.

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Índices de tendencias predichas (0-3)
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return np.zeros(len(X), dtype=int)

            return self.xgb_model.predict(X)

        except Exception as e:
            logger.error(f"Error prediciendo: {str(e)}")
            return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Obtener probabilidades para cada tendencia.

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Probabilidades (n_samples, 4)
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return np.zeros((len(X), 4))

            return self.xgb_model.predict_proba(X)

        except Exception as e:
            logger.error(f"Error prediciendo probabilidades: {str(e)}")
            return np.zeros((len(X), 4))

    # ===========================================
    # INTERPRETACIÓN
    # ===========================================

    def predict_trend_with_confidence(self, X: np.ndarray) -> List[Dict]:
        """
        Predecir tendencia con confianza.

        Args:
            X (np.ndarray): Features

        Retorna:
            List[Dict]: Predicciones con confianza
            [
                {
                    'trend': 'improving',
                    'confidence': 0.85,
                    'probabilities': {'improving': 0.85, 'stable': 0.10, ...}
                },
                ...
            ]
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return []

            predictions = self.predict(X)
            probabilities = self.predict_proba(X)

            results = []

            for pred_idx, probs in zip(predictions, probabilities):
                trend_name = self.trend_labels.get(pred_idx, 'unknown')
                confidence = float(probs[pred_idx])

                # Probabilidades por tendencia
                trend_probs = {
                    self.trend_labels[i]: float(probs[i])
                    for i in range(len(self.trend_labels))
                }

                results.append({
                    'trend': trend_name,
                    'confidence': confidence,
                    'probabilities': trend_probs
                })

            return results

        except Exception as e:
            logger.error(f"Error prediciendo tendencias: {str(e)}")
            return []

    # ===========================================
    # UTILIDADES
    # ===========================================

    def get_trend_distribution(self) -> Dict[str, int]:
        """Obtener distribución de tendencias en datos de entrenamiento."""
        try:
            result = {}
            for trend_idx, count in self.class_distribution.items():
                trend_name = self.trend_labels.get(trend_idx, f"Tendencia {trend_idx}")
                result[trend_name] = count
            return result
        except Exception as e:
            logger.error(f"Error obteniendo distribución: {str(e)}")
            return {}

    def __repr__(self) -> str:
        return (
            f"TrendPredictor(trained={self.is_trained}, "
            f"model={self.xgb_model.__class__.__name__})"
        )
