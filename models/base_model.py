"""
Clase Base Abstracta para Modelos Supervisados
Plataforma Educativa ML

Define la interfaz común para todos los modelos de aprendizaje supervisado:
- PerformancePredictor
- CareerRecommender
- TrendPredictor
- ProgressAnalyzer
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import logging
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("SHAP no está instalado. Instala con: pip install shap")

from shared.config import MODELS_DIR, DEBUG

# Configurar logger
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Clase base abstracta para todos los modelos supervisados.

    Proporciona:
    - Interfaz común para train/predict
    - Manejo de guardado/carga de modelos
    - Evaluación de métricas
    - Logging automático
    """

    def __init__(self, name: str, model_type: str = "supervisado"):
        """
        Inicializar modelo base.

        Args:
            name (str): Nombre único del modelo (ej: "performance_predictor")
            model_type (str): Tipo de modelo (supervisado, no_supervisado, deep_learning)
        """
        self.name = name
        self.model_type = model_type
        self.model = None  # El modelo real (sklearn, xgboost, etc)
        self.is_trained = False
        self.features = []
        self.feature_importance = {}
        self.metadata = {
            'name': name,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'trained': False,
            'trained_at': None,
            'metrics': {}
        }

        logger.info(f"✓ {self.name} inicializado")

    # ===========================================
    # MÉTODOS ABSTRACTOS - DEBE IMPLEMENTARLOS
    # ===========================================

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2,
              **kwargs) -> Dict[str, float]:
        """
        Entrenar el modelo.

        Args:
            X (np.ndarray): Features de entrenamiento (n_samples, n_features)
            y (np.ndarray): Target de entrenamiento (n_samples,)
            validation_split (float): Porcentaje para validación (default 0.2)
            **kwargs: Argumentos adicionales específicos del modelo

        Retorna:
            Dict[str, float]: Diccionario con métricas de entrenamiento
            {
                'train_score': float,
                'val_score': float,
                'metric1': float,
                'metric2': float,
                ...
            }
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realizar predicciones.

        Args:
            X (np.ndarray): Features para predicción (n_samples, n_features)

        Retorna:
            np.ndarray: Predicciones (n_samples,) o (n_samples, n_classes)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Obtener probabilidades de predicción (si aplica).

        Args:
            X (np.ndarray): Features para predicción

        Retorna:
            np.ndarray: Probabilidades (n_samples, n_classes)
        """
        pass

    # ===========================================
    # MÉTODOS DE GUARDADO/CARGA
    # ===========================================

    def save(self, filename: Optional[str] = None, directory: Optional[str] = None) -> str:
        """
        Guardar el modelo entrenado.

        Args:
            filename (str): Nombre del archivo (default: {name}_model.pkl)
            directory (str): Directorio donde guardar (default: MODELS_DIR)

        Retorna:
            str: Ruta completa del archivo guardado
        """
        if not self.is_trained:
            logger.warning(f"{self.name} no está entrenado, salvando de todas formas")

        # Usar defaults si no se proporciona
        if filename is None:
            filename = f"{self.name}_model.pkl"
        if directory is None:
            directory = MODELS_DIR

        filepath = f"{directory}/{filename}"

        try:
            # Actualizar metadata
            self.metadata['trained_at'] = datetime.now().isoformat()
            self.metadata['trained'] = self.is_trained
            self.metadata['file'] = filepath

            # Guardar modelo y metadata
            joblib.dump({
                'model': self.model,
                'features': self.features,
                'feature_importance': self.feature_importance,
                'metadata': self.metadata
            }, filepath)

            logger.info(f"✓ {self.name} guardado en {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"✗ Error guardando {self.name}: {str(e)}")
            raise

    def load(self, filepath: str) -> bool:
        """
        Cargar un modelo entrenado.

        Args:
            filepath (str): Ruta al archivo del modelo

        Retorna:
            bool: True si se cargó exitosamente
        """
        try:
            data = joblib.load(filepath)

            self.model = data['model']
            self.features = data['features']
            self.feature_importance = data.get('feature_importance', {})
            self.metadata = data.get('metadata', {})
            self.is_trained = True

            logger.info(f"✓ {self.name} cargado desde {filepath}")
            return True

        except Exception as e:
            logger.error(f"✗ Error cargando {self.name}: {str(e)}")
            return False

    # ===========================================
    # MÉTODOS DE EVALUACIÓN (CLASIFICACIÓN)
    # ===========================================

    def evaluate_classification(self, y_true: np.ndarray,
                               y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluar modelo de clasificación.

        Args:
            y_true (np.ndarray): Labels verdaderos
            y_pred (np.ndarray): Predicciones del modelo

        Retorna:
            Dict[str, float]: Métricas de clasificación
            {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1': float
            }
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }

            self.metadata['metrics'] = metrics
            if DEBUG:
                logger.info(f"Métricas de {self.name}: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluando clasificación: {str(e)}")
            return {}

    # ===========================================
    # MÉTODOS DE EVALUACIÓN (REGRESIÓN)
    # ===========================================

    def evaluate_regression(self, y_true: np.ndarray,
                           y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluar modelo de regresión.

        Args:
            y_true (np.ndarray): Valores verdaderos
            y_pred (np.ndarray): Predicciones del modelo

        Retorna:
            Dict[str, float]: Métricas de regresión
            {
                'mse': float,
                'rmse': float,
                'mae': float,
                'r2': float
            }
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }

            self.metadata['metrics'] = metrics
            if DEBUG:
                logger.info(f"Métricas de {self.name}: {metrics}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluando regresión: {str(e)}")
            return {}

    # ===========================================
    # UTILIDADES
    # ===========================================

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtener importancia de features (si el modelo lo soporta).

        Retorna:
            Dict[str, float]: Mapeo de feature -> importancia
        """
        return self.feature_importance

    def set_features(self, features: List[str]) -> None:
        """
        Establecer nombres de features.

        Args:
            features (List[str]): Lista de nombres de features
        """
        self.features = features

    def get_metadata(self) -> Dict[str, Any]:
        """
        Obtener metadata del modelo.

        Retorna:
            Dict[str, Any]: Metadata del modelo
        """
        return self.metadata

    # ===========================================
    # MÉTODOS DE VALIDACIÓN CRUZADA (K-FOLD)
    # ===========================================

    def cross_validate_classification(self, X: np.ndarray, y: np.ndarray,
                                     cv: int = 5, stratified: bool = True) -> Dict[str, Any]:
        """
        Realizar validación cruzada para clasificación.

        Args:
            X (np.ndarray): Features de entrenamiento
            y (np.ndarray): Target de entrenamiento
            cv (int): Número de folds (default 5)
            stratified (bool): Si usar StratifiedKFold (mantiene proporciones de clases)

        Retorna:
            Dict[str, Any]: Resultados de validación cruzada
            {
                'accuracy_scores': List[float],
                'precision_scores': List[float],
                'recall_scores': List[float],
                'f1_scores': List[float],
                'mean_accuracy': float,
                'std_accuracy': float,
                'mean_f1': float,
                'std_f1': float
            }
        """
        try:
            # Usar StratifiedKFold para mantener proporciones de clases
            if stratified:
                kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            else:
                kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y if stratified else None)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Entrenar en este fold
                metrics = self.train(X_train, y_train)

                # Predecir en validación
                y_pred = self.predict(X_val)

                # Evaluar
                fold_metrics = self.evaluate_classification(y_val, y_pred)

                accuracy_scores.append(fold_metrics.get('accuracy', 0))
                precision_scores.append(fold_metrics.get('precision', 0))
                recall_scores.append(fold_metrics.get('recall', 0))
                f1_scores.append(fold_metrics.get('f1', 0))

                if DEBUG:
                    logger.info(f"Fold {fold+1}/{cv} - Accuracy: {fold_metrics.get('accuracy', 0):.4f}, F1: {fold_metrics.get('f1', 0):.4f}")

            results = {
                'accuracy_scores': accuracy_scores,
                'precision_scores': precision_scores,
                'recall_scores': recall_scores,
                'f1_scores': f1_scores,
                'mean_accuracy': np.mean(accuracy_scores),
                'std_accuracy': np.std(accuracy_scores),
                'mean_precision': np.mean(precision_scores),
                'std_precision': np.std(precision_scores),
                'mean_recall': np.mean(recall_scores),
                'std_recall': np.std(recall_scores),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'cv_folds': cv
            }

            logger.info(f"✓ Validación Cruzada ({cv}-Fold) completada para {self.name}")
            logger.info(f"  Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            logger.info(f"  F1-Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")

            self.metadata['cross_validation'] = results

            return results

        except Exception as e:
            logger.error(f"Error en validación cruzada: {str(e)}")
            return {}

    def cross_validate_regression(self, X: np.ndarray, y: np.ndarray,
                                 cv: int = 5) -> Dict[str, Any]:
        """
        Realizar validación cruzada para regresión.

        Args:
            X (np.ndarray): Features de entrenamiento
            y (np.ndarray): Target de entrenamiento
            cv (int): Número de folds (default 5)

        Retorna:
            Dict[str, Any]: Resultados de validación cruzada
            {
                'mse_scores': List[float],
                'rmse_scores': List[float],
                'mae_scores': List[float],
                'r2_scores': List[float],
                'mean_mse': float,
                'std_mse': float,
                'mean_r2': float,
                'std_r2': float
            }
        """
        try:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

            mse_scores = []
            rmse_scores = []
            mae_scores = []
            r2_scores = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Entrenar en este fold
                metrics = self.train(X_train, y_train)

                # Predecir en validación
                y_pred = self.predict(X_val)

                # Evaluar
                fold_metrics = self.evaluate_regression(y_val, y_pred)

                mse_scores.append(fold_metrics.get('mse', 0))
                rmse_scores.append(fold_metrics.get('rmse', 0))
                mae_scores.append(fold_metrics.get('mae', 0))
                r2_scores.append(fold_metrics.get('r2', 0))

                if DEBUG:
                    logger.info(f"Fold {fold+1}/{cv} - MSE: {fold_metrics.get('mse', 0):.4f}, R²: {fold_metrics.get('r2', 0):.4f}")

            results = {
                'mse_scores': mse_scores,
                'rmse_scores': rmse_scores,
                'mae_scores': mae_scores,
                'r2_scores': r2_scores,
                'mean_mse': np.mean(mse_scores),
                'std_mse': np.std(mse_scores),
                'mean_rmse': np.mean(rmse_scores),
                'std_rmse': np.std(rmse_scores),
                'mean_mae': np.mean(mae_scores),
                'std_mae': np.std(mae_scores),
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'cv_folds': cv
            }

            logger.info(f"✓ Validación Cruzada ({cv}-Fold) completada para {self.name}")
            logger.info(f"  MSE: {results['mean_mse']:.4f} ± {results['std_mse']:.4f}")
            logger.info(f"  R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")

            self.metadata['cross_validation'] = results

            return results

        except Exception as e:
            logger.error(f"Error en validación cruzada: {str(e)}")
            return {}

    # ===========================================
    # MÉTODOS DE TUNING DE HIPERPARÁMETROS
    # ===========================================

    def hyperparameter_tune(self, X: np.ndarray, y: np.ndarray,
                           param_grid: Dict[str, List[Any]],
                           cv: int = 5, scoring: str = 'accuracy',
                           n_jobs: int = -1) -> Dict[str, Any]:
        """
        Realizar GridSearchCV para tuning automático de hiperparámetros.

        Args:
            X (np.ndarray): Features de entrenamiento
            y (np.ndarray): Target de entrenamiento
            param_grid (Dict): Grid de hiperparámetros a buscar
            cv (int): Número de folds para validación cruzada
            scoring (str): Métrica de scoring ('accuracy', 'f1', 'r2', etc.)
            n_jobs (int): Número de procesadores (-1 para usar todos)

        Retorna:
            Dict[str, Any]: Resultados del tuning
            {
                'best_params': Dict,
                'best_score': float,
                'best_model': estimator,
                'cv_results': Dict
            }
        """
        try:
            if self.model is None:
                logger.error("No hay modelo para realizar tuning")
                return {}

            logger.info(f"Iniciando GridSearchCV con {cv}-Fold CV...")
            logger.info(f"Grid de parámetros: {param_grid}")

            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1 if DEBUG else 0
            )

            grid_search.fit(X, y)

            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_model': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_,
                'n_iter': grid_search.n_iter_,
            }

            logger.info(f"✓ GridSearchCV completado")
            logger.info(f"  Mejores parámetros: {results['best_params']}")
            logger.info(f"  Mejor score: {results['best_score']:.4f}")

            # Actualizar el modelo con los mejores parámetros
            self.model = grid_search.best_estimator_
            self.metadata['hyperparameter_tuning'] = {
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'cv': cv,
                'scoring': scoring
            }

            return results

        except Exception as e:
            logger.error(f"Error en tuning de hiperparámetros: {str(e)}")
            return {}

    def get_cross_validation_results(self) -> Optional[Dict[str, Any]]:
        """
        Obtener resultados de validación cruzada del modelo.

        Retorna:
            Dict o None: Resultados de validación cruzada si están disponibles
        """
        return self.metadata.get('cross_validation', None)

    def get_hyperparameter_tuning_results(self) -> Optional[Dict[str, Any]]:
        """
        Obtener resultados de tuning de hiperparámetros.

        Retorna:
            Dict o None: Resultados de tuning si están disponibles
        """
        return self.metadata.get('hyperparameter_tuning', None)

    # ===========================================
    # MÉTODOS DE EXPLICABILIDAD (SHAP)
    # ===========================================

    def explain_prediction(self, X: np.ndarray, sample_index: int = 0,
                          feature_names: List[str] = None, max_display: int = 10) -> Dict[str, Any]:
        """
        Explicar una predicción individual usando SHAP.

        Args:
            X (np.ndarray): Dataset de features
            sample_index (int): Índice de la muestra a explicar
            feature_names (List[str]): Nombres de features
            max_display (int): Número máximo de features a mostrar

        Retorna:
            Dict: Explicación de la predicción
            {
                'prediction': valor,
                'base_value': valor_base,
                'shap_values': valores_SHAP,
                'feature_contributions': contribuciones_ordenadas,
                'explanation_text': texto_natural
            }
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP no está instalado")
            return {}

        if not self.is_trained:
            logger.error("Modelo no entrenado")
            return {}

        try:
            # Crear explainer basado en tipo de modelo
            if hasattr(self.model, 'predict_proba'):
                # Para clasificación
                explainer = shap.TreeExplainer(self.model)
            else:
                # Para regresión
                explainer = shap.TreeExplainer(self.model)

            # Calcular SHAP values
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = X

            shap_values = explainer.shap_values(X_array)

            # Manejar casos de clasificación multi-clase
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Usar clase positiva

            # Obtener predicción
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(X_array[sample_index:sample_index+1])[0]
            else:
                prediction = self.model.predict(X_array[sample_index:sample_index+1])[0]

            # Nombres de features
            if feature_names is None:
                feature_names = self.features if self.features else [f"feature_{i}" for i in range(X_array.shape[1])]

            # Obtener contribuciones
            sample_shap = shap_values[sample_index]
            feature_impact = [(feature_names[i], sample_shap[i]) for i in range(len(feature_names))]
            feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

            # Tomar top features
            top_features = feature_impact[:max_display]

            # Crear explicación textual
            explanation_text = self._generate_explanation_text(
                prediction, explainer.expected_value, top_features
            )

            result = {
                'prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else prediction.tolist(),
                'base_value': float(explainer.expected_value),
                'shap_values': sample_shap.tolist(),
                'feature_contributions': [
                    {
                        'feature': feature,
                        'contribution': float(contribution),
                        'impact': 'positivo' if contribution > 0 else 'negativo',
                        'magnitude': abs(float(contribution))
                    }
                    for feature, contribution in top_features
                ],
                'explanation_text': explanation_text,
                'top_features': [f[0] for f in top_features],
                'feature_names': feature_names,
            }

            logger.info(f"Explicación SHAP calculada para {self.name}")
            return result

        except Exception as e:
            logger.error(f"Error calculando SHAP values: {str(e)}")
            return {}

    def explain_predictions_batch(self, X: np.ndarray, feature_names: List[str] = None,
                                 max_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Explicar múltiples predicciones usando SHAP.

        Args:
            X (np.ndarray): Dataset de features
            feature_names (List[str]): Nombres de features
            max_samples (int): Número máximo de muestras a explicar

        Retorna:
            List: Lista de explicaciones
        """
        explanations = []

        for i in range(min(max_samples, len(X))):
            exp = self.explain_prediction(X, sample_index=i, feature_names=feature_names)
            if exp:
                explanations.append(exp)

        logger.info(f"Explicaciones SHAP calculadas para {len(explanations)} muestras")
        return explanations

    def get_feature_importance_shap(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Obtener importancia global de features usando SHAP.

        Args:
            X (np.ndarray): Dataset de features
            feature_names (List[str]): Nombres de features

        Retorna:
            Dict: {feature_name: importance_score}
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP no está instalado")
            return {}

        if not self.is_trained:
            logger.error("Modelo no entrenado")
            return {}

        try:
            # Crear explainer
            explainer = shap.TreeExplainer(self.model)

            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = X

            # Calcular SHAP values
            shap_values = explainer.shap_values(X_array)

            # Manejar clasificación multi-clase
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Calcular importancia media
            feature_importance = np.abs(shap_values).mean(axis=0)

            # Nombres de features
            if feature_names is None:
                feature_names = self.features if self.features else [f"feature_{i}" for i in range(X_array.shape[1])]

            # Crear diccionario
            importance_dict = {
                feature_names[i]: float(feature_importance[i])
                for i in range(len(feature_names))
            }

            # Normalizar a suma 100
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: (v / total) * 100 for k, v in importance_dict.items()}

            # Almacenar en metadata
            self.metadata['shap_feature_importance'] = importance_dict

            logger.info(f"Importancia SHAP calculada para {self.name}")
            return importance_dict

        except Exception as e:
            logger.error(f"Error calculando importancia SHAP: {str(e)}")
            return {}

    def _generate_explanation_text(self, prediction: Any, base_value: float,
                                   top_features: List[Tuple[str, float]]) -> str:
        """
        Generar explicación en lenguaje natural.

        Args:
            prediction: Predicción del modelo
            base_value: Valor base (expected value)
            top_features: Top features con contribuciones

        Retorna:
            str: Explicación textual
        """
        if not top_features:
            return "No hay features para explicar"

        explanation = f"Predicción base: {base_value:.2f}\n"
        explanation += f"Predicción final: {prediction:.2f}\n"
        explanation += f"\nFactores principales:\n"

        for feature, contribution in top_features[:3]:
            direction = "aumentó" if contribution > 0 else "disminuyó"
            magnitude = abs(contribution)
            explanation += f"  • {feature}: {direction} la predicción en {magnitude:.4f}\n"

        return explanation.strip()

    def get_shap_summary(self) -> Optional[Dict[str, Any]]:
        """
        Obtener resumen de SHAP almacenado en metadata.

        Retorna:
            Dict: Información SHAP si está disponible
        """
        return {
            'feature_importance': self.metadata.get('shap_feature_importance', {}),
            'shap_available': SHAP_AVAILABLE,
            'model_type': self.model_type
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, trained={self.is_trained})"
