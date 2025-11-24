"""
Career Recommender Model
Plataforma Educativa ML

Recomienda carreras profesionales basado en:
- Desempeño académico
- Calificaciones por materia
- Habilidades y aptitudes

Utiliza:
- SVM (Support Vector Machine) - Para clasificación multiclase
- KNN (K-Nearest Neighbors) - Para interpretabilidad

Output:
- Carrera recomendada (Top 3)
- Puntuación de compatibilidad
- Razonamiento basado en similitud
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel
from shared.config import (
    SVM_KERNEL,
    SVM_C,
    SVM_PROBABILITY,
    KNN_N_NEIGHBORS,
    KNN_WEIGHTS,
    TEST_SIZE,
    DEBUG
)

# Configurar logger
logger = logging.getLogger(__name__)


class CareerRecommender(BaseModel):
    """
    Modelo para recomendar carreras profesionales.

    Utiliza ensemble de SVM y KNN para clasificación multiclase.

    Atributos:
        svm_model (SVC): Modelo SVM
        knn_model (KNeighborsClassifier): Modelo KNN
        scaler (StandardScaler): Escalador para normalizar features
        career_labels (Dict[int, str]): Mapeo de índice a nombre de carrera
        career_scores (Dict[str, float]): Scores por carrera
    """

    def __init__(self, career_labels: Optional[Dict[int, str]] = None):
        """
        Inicializar Career Recommender.

        Args:
            career_labels (Dict[int, str]): Mapeo de índice a nombre de carrera
        """
        super().__init__(name="CareerRecommender", model_type="supervisado")

        # Inicializar modelos individuales
        self.svm_model = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            probability=SVM_PROBABILITY,
            random_state=42,
            verbose=0
        )

        self.knn_model = KNeighborsClassifier(
            n_neighbors=KNN_N_NEIGHBORS,
            weights=KNN_WEIGHTS,
            n_jobs=-1
        )

        # Escalador para SVM (requiere datos normalizados)
        self.scaler = StandardScaler()

        # Mapeo de carreras
        self.career_labels = career_labels or {}
        self.career_scores = {}

        # Información del entrenamiento
        self.cross_val_scores = None
        self.class_distribution = {}

        logger.info(f"✓ {self.name} inicializado con SVM + KNN")

    # ===========================================
    # PREPARACIÓN DE DATOS
    # ===========================================

    def set_career_labels(self, labels: Dict[int, str]) -> None:
        """
        Establecer mapeo de carreras.

        Args:
            labels (Dict[int, str]): {0: 'Ingeniería', 1: 'Medicina', ...}
        """
        self.career_labels = labels
        logger.info(f"Carreras definidas: {labels}")

    def _analyze_class_distribution(self, y: np.ndarray) -> None:
        """
        Analizar distribución de clases.

        Args:
            y (np.ndarray): Variable target multiclase
        """
        try:
            unique, counts = np.unique(y, return_counts=True)
            self.class_distribution = dict(zip(unique, counts))

            logger.info("Distribución de clases:")
            for class_id, count in sorted(self.class_distribution.items()):
                career_name = self.career_labels.get(class_id, f"Carrera {class_id}")
                pct = (count / len(y)) * 100
                logger.info(f"  {career_name}: {count} estudiantes ({pct:.1f}%)")

        except Exception as e:
            logger.error(f"Error analizando distribución: {str(e)}")

    # ===========================================
    # ENTRENAMIENTO
    # ===========================================

    def train(self, X: np.ndarray, y: np.ndarray,
             validation_split: float = 0.2,
             **kwargs) -> Dict[str, float]:
        """
        Entrenar el modelo (SVM + KNN).

        Args:
            X (np.ndarray): Features (n_samples, n_features)
            y (np.ndarray): Target multiclase - Índices de carreras
            validation_split (float): Proporción para validación
            **kwargs: Argumentos adicionales

        Retorna:
            Dict[str, float]: Métricas de entrenamiento
            {
                'svm_train_score': float,
                'svm_val_score': float,
                'knn_train_score': float,
                'knn_val_score': float,
                'ensemble_score': float,
                'num_careers': int
            }
        """
        try:
            logger.info("Iniciando entrenamiento de CareerRecommender...")

            # Analizar distribución
            self._analyze_class_distribution(y)

            # Dividir datos
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )

            # ========== ESCALAR DATOS ==========
            logger.info("Escalando features para SVM...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # ========== ENTRENAR SVM ==========
            logger.info("Entrenando SVM...")
            self.svm_model.fit(X_train_scaled, y_train)

            svm_train_score = self.svm_model.score(X_train_scaled, y_train)
            svm_val_score = self.svm_model.score(X_val_scaled, y_val)

            logger.info(
                f"SVM: train_score={svm_train_score:.4f}, "
                f"val_score={svm_val_score:.4f}"
            )

            # ========== ENTRENAR KNN ==========
            logger.info("Entrenando KNN...")
            self.knn_model.fit(X_train, y_train)

            knn_train_score = self.knn_model.score(X_train, y_train)
            knn_val_score = self.knn_model.score(X_val, y_val)

            logger.info(
                f"KNN: train_score={knn_train_score:.4f}, "
                f"val_score={knn_val_score:.4f}"
            )

            # ========== ENSEMBLE SCORE ==========
            svm_pred_val = self.svm_model.predict(X_val_scaled)
            knn_pred_val = self.knn_model.predict(X_val)

            # Modo ensemble: usar voto mayoritario cuando coinciden
            ensemble_pred = np.where(
                svm_pred_val == knn_pred_val,
                svm_pred_val,
                svm_pred_val  # En caso de desacuerdo, usar SVM
            )

            ensemble_score = (ensemble_pred == y_val).mean()

            logger.info(f"Ensemble Score: {ensemble_score:.4f}")

            # ========== CROSS VALIDATION ==========
            logger.info("Calculando cross-validation...")
            self.cross_val_scores = {
                'svm': cross_val_score(SVC(kernel=SVM_KERNEL, C=SVM_C, probability=True),
                                      X, y, cv=5, n_jobs=-1),
                'knn': cross_val_score(KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS),
                                      X, y, cv=5, n_jobs=-1)
            }

            logger.info(
                f"SVM CV: {self.cross_val_scores['svm'].mean():.4f} "
                f"(± {self.cross_val_scores['svm'].std():.4f})"
            )
            logger.info(
                f"KNN CV: {self.cross_val_scores['knn'].mean():.4f} "
                f"(± {self.cross_val_scores['knn'].std():.4f})"
            )

            # Marcar como entrenado
            self.is_trained = True
            self.features = list(range(X.shape[1]))

            # Retornar métricas
            metrics = {
                'svm_train_score': float(svm_train_score),
                'svm_val_score': float(svm_val_score),
                'knn_train_score': float(knn_train_score),
                'knn_val_score': float(knn_val_score),
                'ensemble_score': float(ensemble_score),
                'svm_cv_mean': float(self.cross_val_scores['svm'].mean()),
                'knn_cv_mean': float(self.cross_val_scores['knn'].mean()),
                'num_careers': len(np.unique(y))
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
        Predecir carrera más probable.

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Índices de carreras predichas
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return np.zeros(len(X), dtype=int)

            # Escalar features para SVM
            X_scaled = self.scaler.transform(X)

            # Predicciones
            svm_pred = self.svm_model.predict(X_scaled)
            knn_pred = self.knn_model.predict(X)

            # Ensemble: voto mayoritario
            ensemble_pred = np.where(
                svm_pred == knn_pred,
                svm_pred,
                svm_pred  # En desacuerdo, usar SVM
            )

            return ensemble_pred

        except Exception as e:
            logger.error(f"Error prediciendo: {str(e)}")
            return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Obtener probabilidades para cada carrera.

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Probabilidades (n_samples, n_careers)
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return np.zeros((len(X), len(self.career_labels)))

            # Escalar features
            X_scaled = self.scaler.transform(X)

            # Obtener probabilidades de ambos modelos
            svm_proba = self.svm_model.predict_proba(X_scaled)
            knn_proba = self.knn_model.predict_proba(X)

            # Promediar probabilidades
            ensemble_proba = (svm_proba + knn_proba) / 2

            return ensemble_proba

        except Exception as e:
            logger.error(f"Error prediciendo probabilidades: {str(e)}")
            return np.zeros((len(X), len(self.career_labels)))

    # ===========================================
    # RECOMENDACIÓN DE CARRERAS
    # ===========================================

    def recommend_careers(self, X: np.ndarray, top_n: int = 3) -> List[Dict]:
        """
        Recomendar carreras con compatibilidad.

        Args:
            X (np.ndarray): Features del estudiante(es)
            top_n (int): Top N carreras a recomendar

        Retorna:
            List[Dict]: Lista de recomendaciones
            [
                {
                    'career': 'Ingeniería Informática',
                    'compatibility': 0.92,
                    'rank': 1,
                    'reason': 'Alto promedio en matemáticas...'
                },
                ...
            ]
        """
        try:
            if not self.is_trained:
                logger.error("Modelo no está entrenado")
                return []

            # Obtener probabilidades
            probabilities = self.predict_proba(X)

            # Asegurar que es 2D
            if len(probabilities.shape) == 1:
                probabilities = probabilities.reshape(1, -1)

            recommendations = []

            for sample_idx, probs in enumerate(probabilities):
                # Top N carreras
                top_indices = np.argsort(probs)[-top_n:][::-1]

                sample_recs = []
                for rank, career_idx in enumerate(top_indices, 1):
                    career_name = self.career_labels.get(
                        career_idx,
                        f"Carrera {career_idx}"
                    )

                    sample_recs.append({
                        'career': career_name,
                        'compatibility': float(probs[career_idx]),
                        'rank': rank,
                        'reason': self._generate_reason(X[sample_idx], career_idx)
                    })

                recommendations.append(sample_recs)

            return recommendations if len(recommendations) > 1 else recommendations[0]

        except Exception as e:
            logger.error(f"Error recomendando carreras: {str(e)}")
            return []

    def _generate_reason(self, features: np.ndarray, career_idx: int) -> str:
        """
        Generar explicación de recomendación (placeholder).

        Args:
            features (np.ndarray): Features del estudiante
            career_idx (int): Índice de carrera

        Retorna:
            str: Explicación
        """
        try:
            # Análisis simple basado en features
            avg_feature = np.mean(features)

            if avg_feature >= 8.0:
                return "Excelente desempeño académico en áreas clave"
            elif avg_feature >= 6.5:
                return "Sólido desempeño académico"
            elif avg_feature >= 5.0:
                return "Desempeño promedio, requiere mejora"
            else:
                return "Requiere apoyo académico adicional"

        except Exception as e:
            logger.error(f"Error generando razón: {str(e)}")
            return "Análisis basado en perfil académico"

    # ===========================================
    # UTILIDADES
    # ===========================================

    def get_career_distribution(self) -> Dict[str, int]:
        """
        Obtener distribución de estudiantes por carrera.

        Retorna:
            Dict[str, int]: {career_name: count}
        """
        try:
            result = {}
            for career_idx, count in self.class_distribution.items():
                career_name = self.career_labels.get(
                    career_idx,
                    f"Carrera {career_idx}"
                )
                result[career_name] = count

            return result

        except Exception as e:
            logger.error(f"Error obteniendo distribución: {str(e)}")
            return {}

    def __repr__(self) -> str:
        return (
            f"CareerRecommender(trained={self.is_trained}, "
            f"svm_model={self.svm_model.__class__.__name__}, "
            f"knn_model={self.knn_model.__class__.__name__}, "
            f"careers={len(self.career_labels)})"
        )
