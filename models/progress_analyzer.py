"""
Progress Analyzer Model
Plataforma Educativa ML

Analiza y predice el progreso académico de estudiantes usando:
- Regresión Lineal: Para tendencias básicas
- Regresión Polinomial: Para patrones complejos

Output:
- Predicción de calificación futura
- Velocidad de aprendizaje
- Proyección a fin de período
- Puntos de inflexión
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

from .base_model import BaseModel
from shared.config import TEST_SIZE, RANDOM_STATE, DEBUG

# Configurar logger
logger = logging.getLogger(__name__)


class ProgressAnalyzer(BaseModel):
    """
    Modelo para analizar y predecir progreso académico.

    Utiliza regresión lineal y polinomial para análisis temporal.

    Atributos:
        linear_model: Modelo de regresión lineal
        polynomial_model: Modelo de regresión polinomial
        poly_features: Transformador polinomial
        learning_rates: Velocidad de aprendizaje por estudiante
    """

    def __init__(self, polynomial_degree: int = 2):
        """
        Inicializar Progress Analyzer.

        Args:
            polynomial_degree (int): Grado del polinomio (default 2 = cuadrático)
        """
        super().__init__(name="ProgressAnalyzer", model_type="supervisado")

        self.polynomial_degree = polynomial_degree

        # Modelos
        self.linear_model = LinearRegression()
        self.polynomial_model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=polynomial_degree)

        # Estadísticas
        self.learning_rates = {}
        self.student_projections = {}
        self.inflection_points = {}

        logger.info(f"✓ {self.name} inicializado (polinomio grado {polynomial_degree})")

    # ===========================================
    # ANÁLISIS DE PROGRESO
    # ===========================================

    @staticmethod
    def calculate_learning_rate(grades_sequence: np.ndarray) -> float:
        """
        Calcular velocidad de aprendizaje (pendiente promedio).

        Args:
            grades_sequence (np.ndarray): Secuencia de calificaciones

        Retorna:
            float: Velocidad de aprendizaje (puntos por período)
        """
        try:
            if len(grades_sequence) < 2:
                return 0.0

            # Regresión lineal simple
            x = np.arange(len(grades_sequence))
            slope = np.polyfit(x, grades_sequence, 1)[0]

            return float(slope)

        except Exception as e:
            logger.warning(f"Error calculando learning rate: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_acceleration(grades_sequence: np.ndarray) -> float:
        """
        Calcular aceleración (cambio de velocidad).

        Args:
            grades_sequence (np.ndarray): Secuencia de calificaciones

        Retorna:
            float: Aceleración (positiva = mejorando más rápido)
        """
        try:
            if len(grades_sequence) < 3:
                return 0.0

            # Diferencias de primer orden (velocidad)
            diff1 = np.diff(grades_sequence)

            # Diferencias de segundo orden (aceleración)
            if len(diff1) >= 2:
                acceleration = diff1[-1] - diff1[0]
                return float(acceleration)

            return 0.0

        except Exception as e:
            logger.warning(f"Error calculando aceleración: {str(e)}")
            return 0.0

    @staticmethod
    def find_inflection_points(grades_sequence: np.ndarray,
                              threshold: float = 0.5) -> List[int]:
        """
        Encontrar puntos de inflexión en la secuencia de calificaciones.

        Un punto de inflexión es donde la tendencia cambia significativamente.

        Args:
            grades_sequence (np.ndarray): Secuencia de calificaciones
            threshold (float): Cambio mínimo para considerar inflexión

        Retorna:
            List[int]: Índices de puntos de inflexión
        """
        try:
            if len(grades_sequence) < 3:
                return []

            diff = np.diff(grades_sequence)
            inflection_points = []

            for i in range(1, len(diff)):
                # Cambio de signo en la derivada = inflexión
                if diff[i-1] * diff[i] < 0:
                    inflection_points.append(i)
                # O cambio significativo de magnitud
                elif abs(diff[i] - diff[i-1]) > threshold:
                    inflection_points.append(i)

            return inflection_points

        except Exception as e:
            logger.warning(f"Error encontrando inflexiones: {str(e)}")
            return []

    # ===========================================
    # PREPARACIÓN DE DATOS
    # ===========================================

    def _prepare_regression_data(self, X: np.ndarray, y: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preparar datos para regresión (agregar columna temporal).

        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target (calificaciones)

        Retorna:
            Tuple: (X_with_time, X_val_with_time, y, y_val)
        """
        try:
            # Agregar columna de tiempo (índice secuencial)
            time_col = np.arange(len(X)).reshape(-1, 1)
            X_with_time = np.hstack([X, time_col])

            return X_with_time, y

        except Exception as e:
            logger.error(f"Error preparando datos: {str(e)}")
            return X, y

    # ===========================================
    # ENTRENAMIENTO
    # ===========================================

    def train(self, X: np.ndarray, y: np.ndarray,
             validation_split: float = 0.2,
             **kwargs) -> Dict[str, float]:
        """
        Entrenar modelos de regresión.

        Args:
            X (np.ndarray): Features (n_samples, n_features)
            y (np.ndarray): Target - Calificaciones a predecir
            validation_split (float): Proporción para validación
            **kwargs: Argumentos adicionales

        Retorna:
            Dict[str, float]: Métricas de entrenamiento
            {
                'linear_train_r2': float,
                'linear_val_r2': float,
                'polynomial_train_r2': float,
                'polynomial_val_r2': float,
                'best_model': str
            }
        """
        try:
            logger.info("Iniciando entrenamiento de ProgressAnalyzer...")

            # Dividir datos
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=RANDOM_STATE
            )

            # ========== REGRESIÓN LINEAL ==========
            logger.info("Entrenando regresión lineal...")
            self.linear_model.fit(X_train, y_train)

            linear_train_r2 = self.linear_model.score(X_train, y_train)
            linear_val_r2 = self.linear_model.score(X_val, y_val)
            linear_train_rmse = np.sqrt(mean_squared_error(y_train, self.linear_model.predict(X_train)))
            linear_val_rmse = np.sqrt(mean_squared_error(y_val, self.linear_model.predict(X_val)))

            logger.info(
                f"Regresión Lineal: "
                f"train_r2={linear_train_r2:.4f}, val_r2={linear_val_r2:.4f}, "
                f"train_rmse={linear_train_rmse:.4f}, val_rmse={linear_val_rmse:.4f}"
            )

            # ========== REGRESIÓN POLINOMIAL ==========
            logger.info(f"Entrenando regresión polinomial (grado {self.polynomial_degree})...")

            # Transformar features a polinomiales
            X_train_poly = self.poly_features.fit_transform(X_train)
            X_val_poly = self.poly_features.transform(X_val)

            self.polynomial_model.fit(X_train_poly, y_train)

            poly_train_r2 = self.polynomial_model.score(X_train_poly, y_train)
            poly_val_r2 = self.polynomial_model.score(X_val_poly, y_val)
            poly_train_rmse = np.sqrt(mean_squared_error(y_train, self.polynomial_model.predict(X_train_poly)))
            poly_val_rmse = np.sqrt(mean_squared_error(y_val, self.polynomial_model.predict(X_val_poly)))

            logger.info(
                f"Regresión Polinomial: "
                f"train_r2={poly_train_r2:.4f}, val_r2={poly_val_r2:.4f}, "
                f"train_rmse={poly_train_rmse:.4f}, val_rmse={poly_val_rmse:.4f}"
            )

            # ========== SELECCIONAR MEJOR MODELO ==========
            best_model = "polynomial" if poly_val_r2 > linear_val_r2 else "linear"
            logger.info(f"Mejor modelo: {best_model}")

            # Marcar como entrenado
            self.is_trained = True
            self.features = list(range(X.shape[1]))

            # Retornar métricas
            metrics = {
                'linear_train_r2': float(linear_train_r2),
                'linear_val_r2': float(linear_val_r2),
                'linear_train_rmse': float(linear_train_rmse),
                'linear_val_rmse': float(linear_val_rmse),
                'polynomial_train_r2': float(poly_train_r2),
                'polynomial_val_r2': float(poly_val_r2),
                'polynomial_train_rmse': float(poly_train_rmse),
                'polynomial_val_rmse': float(poly_val_rmse),
                'best_model': best_model
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

    def predict(self, X: np.ndarray, use_polynomial: bool = True) -> np.ndarray:
        """
        Predecir calificaciones futuras.

        Args:
            X (np.ndarray): Features
            use_polynomial (bool): Si True, usar modelo polinomial

        Retorna:
            np.ndarray: Predicciones de calificaciones
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return np.zeros(len(X))

            if use_polynomial:
                X_poly = self.poly_features.transform(X)
                return self.polynomial_model.predict(X_poly)
            else:
                return self.linear_model.predict(X)

        except Exception as e:
            logger.error(f"Error prediciendo: {str(e)}")
            return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Para regresión, retorna intervalo de confianza (simplificado).

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Predicciones (para compatibilidad con BaseModel)
        """
        # Para regresión, predict_proba no aplica, pero lo mantenemos para compatibilidad
        return self.predict(X).reshape(-1, 1)

    # ===========================================
    # ANÁLISIS Y PROYECCIONES
    # ===========================================

    def project_progress(self, X: np.ndarray, periods_ahead: int = 5,
                        use_polynomial: bool = True) -> List[Dict]:
        """
        Proyectar progreso hacia el futuro.

        Args:
            X (np.ndarray): Features actuales
            periods_ahead (int): Períodos a proyectar
            use_polynomial (bool): Usar modelo polinomial

        Retorna:
            List[Dict]: Proyecciones
            [
                {
                    'current_grade': 7.5,
                    'projected_grade': 8.2,
                    'learning_rate': 0.14,
                    'acceleration': 0.05,
                    'trajectory': 'improving',
                    'periods_ahead': 5
                },
                ...
            ]
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return []

            # Predecir para período actual
            current_predictions = self.predict(X, use_polynomial=use_polynomial)

            # Crear features para períodos futuros (simplificado: agregar offset temporal)
            projections = []

            for current_pred in current_predictions:
                # Crear característica de cambio temporal
                time_offset = np.array([[periods_ahead]])  # Asumir una feature de tiempo

                # Estimar cambio basado en modelo
                if len(X) > 0:
                    avg_feature = np.mean(X)
                    future_features = X[0:1] + (time_offset * 0.1)  # Ajuste simplificado
                    future_pred = self.predict(future_features, use_polynomial=use_polynomial)[0]
                else:
                    future_pred = current_pred

                learning_rate = self.calculate_learning_rate(np.array([current_pred, future_pred]))

                if learning_rate > 0.1:
                    trajectory = "improving"
                elif learning_rate < -0.1:
                    trajectory = "declining"
                else:
                    trajectory = "stable"

                projections.append({
                    'current_grade': float(current_pred),
                    'projected_grade': float(future_pred),
                    'learning_rate': float(learning_rate),
                    'trajectory': trajectory,
                    'periods_ahead': periods_ahead
                })

            return projections

        except Exception as e:
            logger.error(f"Error proyectando progreso: {str(e)}")
            return []

    def analyze_student_progress(self, student_grades: np.ndarray) -> Dict:
        """
        Analizar progreso completo de un estudiante.

        Args:
            student_grades (np.ndarray): Secuencia de calificaciones

        Retorna:
            Dict: Análisis completo
            {
                'mean_grade': float,
                'max_grade': float,
                'min_grade': float,
                'learning_rate': float,
                'acceleration': float,
                'inflection_points': list,
                'status': str
            }
        """
        try:
            learning_rate = self.calculate_learning_rate(student_grades)
            acceleration = self.calculate_acceleration(student_grades)
            inflections = self.find_inflection_points(student_grades)

            # Determinar estado
            if learning_rate > 0.2:
                status = "rapid_improvement"
            elif learning_rate > 0.05:
                status = "steady_improvement"
            elif learning_rate < -0.2:
                status = "significant_decline"
            elif learning_rate < -0.05:
                status = "gradual_decline"
            else:
                status = "stable"

            return {
                'mean_grade': float(np.mean(student_grades)),
                'max_grade': float(np.max(student_grades)),
                'min_grade': float(np.min(student_grades)),
                'std_dev': float(np.std(student_grades)),
                'learning_rate': float(learning_rate),
                'acceleration': float(acceleration),
                'inflection_points': inflections,
                'status': status,
                'num_observations': len(student_grades)
            }

        except Exception as e:
            logger.error(f"Error analizando progreso: {str(e)}")
            return {}

    # ===========================================
    # UTILIDADES
    # ===========================================

    def get_model_coefficients(self) -> Dict[str, List[float]]:
        """
        Obtener coeficientes de los modelos.

        Retorna:
            Dict: Coeficientes por modelo
        """
        try:
            return {
                'linear_coef': self.linear_model.coef_.tolist(),
                'linear_intercept': float(self.linear_model.intercept_),
                'polynomial_coef': self.polynomial_model.coef_.tolist(),
                'polynomial_intercept': float(self.polynomial_model.intercept_)
            }
        except Exception as e:
            logger.error(f"Error obteniendo coeficientes: {str(e)}")
            return {}

    def __repr__(self) -> str:
        return (
            f"ProgressAnalyzer(trained={self.is_trained}, "
            f"poly_degree={self.polynomial_degree})"
        )
