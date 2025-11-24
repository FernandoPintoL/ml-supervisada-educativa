"""
Performance Predictor Model
Plataforma Educativa ML

Predice el riesgo de bajo desempeño de estudiantes usando:
- Random Forest (para interpretabilidad)
- XGBoost (para precisión)

Output:
- Riesgo Alto (> 70%)
- Riesgo Medio (40-70%)
- Riesgo Bajo (< 40%)
"""

import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb

from .base_model import BaseModel
from shared.config import (
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    RF_RANDOM_STATE,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_RANDOM_STATE,
    PERFORMANCE_RISK_THRESHOLD_HIGH,
    PERFORMANCE_RISK_THRESHOLD_MEDIUM,
    TEST_SIZE,
    DEBUG
)

# Configurar logger
logger = logging.getLogger(__name__)


class PerformancePredictor(BaseModel):
    """
    Modelo para predecir riesgo de bajo desempeño en estudiantes.

    Utiliza ensemble de Random Forest y XGBoost.

    Atributos:
        rf_model (RandomForestClassifier): Modelo Random Forest
        xgb_model (XGBClassifier): Modelo XGBoost
        threshold_high (float): Umbral para riesgo alto
        threshold_medium (float): Umbral para riesgo medio
    """

    def __init__(self):
        """Inicializar Performance Predictor."""
        super().__init__(name="PerformancePredictor", model_type="supervisado")

        # Inicializar modelos individuales
        self.rf_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=XGB_RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            base_score=0.5,  # Ensure valid base score for logistic loss
            scale_pos_weight=1  # Handle class balance
        )

        # Thresholds para clasificación de riesgo
        self.threshold_high = PERFORMANCE_RISK_THRESHOLD_HIGH
        self.threshold_medium = PERFORMANCE_RISK_THRESHOLD_MEDIUM

        # Almacenar información del entrenamiento
        self.rf_importance = {}
        self.xgb_importance = {}
        self.cross_val_scores = None
        self.xgb_trained = False  # Flag para saber si XGBoost se entrenó exitosamente

        logger.info(f"✓ {self.name} inicializado con RF + XGBoost")

    # ===========================================
    # PREPARACIÓN DE DATOS
    # ===========================================

    def _create_binary_target(self, df: pd.DataFrame,
                             grade_column: str = 'promedio_ultimas_notas',
                             threshold: float = 3.0) -> np.ndarray:
        """
        Crear variable binaria de riesgo a partir de calificaciones.

        Args:
            df (DataFrame): Datos con calificaciones
            grade_column (str): Nombre de columna con promedio de notas
            threshold (float): Nota mínima para considerarse "sin riesgo"

        Retorna:
            np.ndarray: Array binario [0=riesgo, 1=sin riesgo]
        """
        try:
            if grade_column not in df.columns:
                logger.error(f"Columna {grade_column} no encontrada")
                return np.zeros(len(df))

            # 1 = sin riesgo (nota >= threshold), 0 = en riesgo
            y = (df[grade_column] >= threshold).astype(int).values

            num_risk = (y == 0).sum()
            num_ok = (y == 1).sum()

            logger.info(
                f"Variable target creada: "
                f"{num_ok} sin riesgo, {num_risk} en riesgo"
            )

            return y

        except Exception as e:
            logger.error(f"Error creando variable target: {str(e)}")
            return np.zeros(len(df))

    # ===========================================
    # ENTRENAMIENTO
    # ===========================================

    def train(self, X: np.ndarray, y: np.ndarray,
             validation_split: float = 0.2,
             **kwargs) -> Dict[str, float]:
        """
        Entrenar el modelo (Random Forest + XGBoost).

        Args:
            X (np.ndarray): Features (n_samples, n_features)
            y (np.ndarray): Target binario (n_samples,)
            validation_split (float): Proporción para validación
            **kwargs: Argumentos adicionales

        Retorna:
            Dict[str, float]: Métricas de entrenamiento
            {
                'rf_train_score': float,
                'rf_val_score': float,
                'xgb_train_score': float,
                'xgb_val_score': float,
                'ensemble_score': float
            }
        """
        try:
            logger.info("Iniciando entrenamiento de PerformancePredictor...")

            # Dividir datos
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )

            # ========== ENTRENAR RANDOM FOREST ==========
            logger.info("Entrenando Random Forest...")
            self.rf_model.fit(X_train, y_train)

            rf_train_score = self.rf_model.score(X_train, y_train)
            rf_val_score = self.rf_model.score(X_val, y_val)

            # Extraer importancia de features
            self.rf_importance = dict(zip(
                range(X.shape[1]),
                self.rf_model.feature_importances_
            ))

            logger.info(
                f"Random Forest: train_score={rf_train_score:.4f}, "
                f"val_score={rf_val_score:.4f}"
            )

            # ========== ENTRENAR XGBOOST ==========
            logger.info("Entrenando XGBoost...")

            # Verificar si hay solo una clase en training data
            unique_classes_train = np.unique(y_train)
            xgb_train_score = 0.0
            xgb_val_score = 0.0

            if len(unique_classes_train) < 2:
                logger.warning(
                    f"⚠ Solo una clase en training data ({unique_classes_train}). "
                    f"XGBoost requiere dos clases. Usando solo Random Forest."
                )
                self.xgb_trained = False
            else:
                try:
                    self.xgb_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )

                    xgb_train_score = self.xgb_model.score(X_train, y_train)
                    xgb_val_score = self.xgb_model.score(X_val, y_val)
                    self.xgb_trained = True
                except ValueError as e:
                    logger.warning(
                        f"⚠ XGBoost falló ({str(e)}). "
                        f"Usando solo Random Forest para predicciones."
                    )
                    self.xgb_trained = False

            # Extraer importancia de features (solo si XGBoost se entrenó)
            if self.xgb_trained:
                self.xgb_importance = dict(zip(
                    range(X.shape[1]),
                    self.xgb_model.feature_importances_
                ))

                logger.info(
                    f"XGBoost: train_score={xgb_train_score:.4f}, "
                    f"val_score={xgb_val_score:.4f}"
                )
            else:
                logger.info("XGBoost: no entrenado (usar solo Random Forest)")

            # ========== ENSEMBLE SCORE ==========
            # Promediar predicciones de ambos modelos
            # Manejar casos donde predict_proba solo devuelve 1 columna (una sola clase)
            rf_proba = self.rf_model.predict_proba(X_val)

            if self.xgb_trained:
                xgb_proba = self.xgb_model.predict_proba(X_val)
                rf_pred_val = rf_proba[:, 1] if rf_proba.shape[1] > 1 else (1 - rf_proba[:, 0])
                xgb_pred_val = xgb_proba[:, 1] if xgb_proba.shape[1] > 1 else (1 - xgb_proba[:, 0])
                ensemble_pred = (rf_pred_val + xgb_pred_val) / 2
            else:
                # Solo usar Random Forest si XGBoost no se entrenó
                ensemble_pred = rf_proba[:, 1] if rf_proba.shape[1] > 1 else (1 - rf_proba[:, 0])

            ensemble_pred_binary = (ensemble_pred >= 0.5).astype(int)
            ensemble_score = (ensemble_pred_binary == y_val).mean()

            logger.info(f"Ensemble Score: {ensemble_score:.4f}")

            # ========== CROSS VALIDATION ==========
            logger.info("Calculando cross-validation...")
            self.cross_val_scores = {
                'rf': cross_val_score(self.rf_model, X, y, cv=5, n_jobs=-1),
            }

            logger.info(
                f"RF CV: {self.cross_val_scores['rf'].mean():.4f} "
                f"(± {self.cross_val_scores['rf'].std():.4f})"
            )

            if self.xgb_trained:
                self.cross_val_scores['xgb'] = cross_val_score(self.xgb_model, X, y, cv=5, n_jobs=-1)
                logger.info(
                    f"XGB CV: {self.cross_val_scores['xgb'].mean():.4f} "
                    f"(± {self.cross_val_scores['xgb'].std():.4f})"
                )

            # Marcar como entrenado
            self.is_trained = True
            self.feature_importance = self.rf_importance  # Usar RF como referencia

            # Retornar métricas
            metrics = {
                'rf_train_score': float(rf_train_score),
                'rf_val_score': float(rf_val_score),
                'xgb_train_score': float(xgb_train_score),
                'xgb_val_score': float(xgb_val_score),
                'ensemble_score': float(ensemble_score),
                'rf_cv_mean': float(self.cross_val_scores['rf'].mean()),
                'xgb_trained': self.xgb_trained,
                'xgb_cv_mean': float(self.cross_val_scores['xgb'].mean()) if 'xgb' in self.cross_val_scores else 0.0
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
        Predecir riesgo (clase binaria).

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Predicción binaria [0=riesgo, 1=sin riesgo]
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return np.zeros(len(X))

            # Si XGBoost no fue entrenado, usar solo Random Forest
            if not self.xgb_trained:
                logger.debug("Usando solo Random Forest (XGBoost no entrenado)")
                return self.rf_model.predict(X)

            # Usar ambos modelos y promediar
            rf_pred = self.rf_model.predict(X)
            xgb_pred = self.xgb_model.predict(X)

            # Ensemble: promediar y redondear
            ensemble_pred = (rf_pred + xgb_pred) / 2
            return (ensemble_pred >= 0.5).astype(int)

        except Exception as e:
            logger.error(f"Error prediciendo: {str(e)}")
            return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Obtener probabilidades de riesgo.

        Retorna probabilidad de SIN RIESGO (clase 1).

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Probabilidades de sin riesgo (0-1)
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return np.zeros(len(X))

            # Si XGBoost no fue entrenado, usar solo Random Forest
            if not self.xgb_trained:
                logger.debug("Usando solo Random Forest (XGBoost no entrenado)")
                rf_proba_raw = self.rf_model.predict_proba(X)
                rf_proba = rf_proba_raw[:, 1] if rf_proba_raw.shape[1] > 1 else (1 - rf_proba_raw[:, 0])
                return rf_proba.reshape(-1, 1)

            # Obtener probabilidades de ambos modelos
            # Manejar casos donde predict_proba solo devuelve 1 columna
            rf_proba_raw = self.rf_model.predict_proba(X)
            xgb_proba_raw = self.xgb_model.predict_proba(X)

            rf_proba = rf_proba_raw[:, 1] if rf_proba_raw.shape[1] > 1 else (1 - rf_proba_raw[:, 0])
            xgb_proba = xgb_proba_raw[:, 1] if xgb_proba_raw.shape[1] > 1 else (1 - xgb_proba_raw[:, 0])

            # Promediar probabilidades
            ensemble_proba = (rf_proba + xgb_proba) / 2

            return ensemble_proba.reshape(-1, 1)

        except Exception as e:
            logger.error(f"Error prediciendo probabilidades: {str(e)}")
            return np.zeros((len(X), 1))

    # ===========================================
    # INTERPRETACIÓN DE RIESGO
    # ===========================================

    def predict_risk_level(self, X: np.ndarray) -> list:
        """
        Predecir nivel de riesgo (Alto, Medio, Bajo).

        Args:
            X (np.ndarray): Features

        Retorna:
            List[Dict]: Lista de riesgos
            [
                {'risk_level': 'High', 'risk_score': 0.85, 'status': 'critical'},
                {'risk_level': 'Medium', 'risk_score': 0.55, 'status': 'warning'},
                {'risk_level': 'Low', 'risk_score': 0.25, 'status': 'ok'},
                ...
            ]
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return []

            # Obtener probabilidades de RIESGO (clase 0)
            probabilities = 1 - self.predict_proba(X).flatten()

            results = []
            for prob in probabilities:
                if prob >= self.threshold_high:
                    risk_level = 'High'
                    status = 'critical'
                elif prob >= self.threshold_medium:
                    risk_level = 'Medium'
                    status = 'warning'
                else:
                    risk_level = 'Low'
                    status = 'ok'

                results.append({
                    'risk_level': risk_level,
                    'risk_score': float(prob),
                    'status': status
                })

            return results

        except Exception as e:
            logger.error(f"Error prediciendo nivel de riesgo: {str(e)}")
            return []

    # ===========================================
    # FEATURE IMPORTANCE
    # ===========================================

    def get_top_features(self, n: int = 10) -> dict:
        """
        Obtener features más importantes.

        Args:
            n (int): Número de features a retornar

        Retorna:
            Dict: Features ordenadas por importancia
            {
                'rf': {feature_name: importance, ...},
                'xgb': {feature_name: importance, ...}
            }
        """
        try:
            rf_sorted = sorted(
                self.rf_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n]

            xgb_sorted = sorted(
                self.xgb_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n]

            return {
                'rf': dict(rf_sorted),
                'xgb': dict(xgb_sorted)
            }

        except Exception as e:
            logger.error(f"Error obteniendo top features: {str(e)}")
            return {}

    def __repr__(self) -> str:
        return (
            f"PerformancePredictor(trained={self.is_trained}, "
            f"rf_model={self.rf_model.__class__.__name__}, "
            f"xgb_model={self.xgb_model.__class__.__name__})"
        )
