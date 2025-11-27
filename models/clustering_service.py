"""
K-Means Clustering Service para Plataforma Educativa

Agrupa estudiantes en clusters basados en su desempeño para:
- Identificar patrones de aprendizaje
- Detectar anomalías (estudiantes que no coinciden con su cluster)
- Validar coherencia de predicciones contra cluster esperado
"""

import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

logger = logging.getLogger(__name__)

class ClusteringService:
    """
    Servicio de clustering para agrupar estudiantes

    Usa K-Means para crear 3 clusters:
    - Cluster 0: Bajo Desempeño (High Risk)
    - Cluster 1: Desempeño Medio (Medium Risk)
    - Cluster 2: Alto Desempeño (Low Risk)
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Inicializar servicio de clustering

        Args:
            n_clusters: Número de clusters (por defecto 3)
            random_state: Seed para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.scaler = None
        self.feature_names = None
        self.cluster_risk_mapping = {
            0: 'alto',    # Bajo desempeño = Alto riesgo
            1: 'medio',   # Desempeño medio = Riesgo medio
            2: 'bajo',    # Alto desempeño = Bajo riesgo
        }
        self.cluster_names = {
            0: 'Bajo Desempeño',
            1: 'Desempeño Medio',
            2: 'Alto Desempeño',
        }

    def train_clustering(self, features_df: pd.DataFrame, student_ids: List[int] = None) -> Dict:
        """
        Entrenar modelo K-Means con features de estudiantes

        Args:
            features_df: DataFrame con features de estudiantes
            student_ids: IDs de estudiantes (opcional)

        Returns:
            Diccionario con métricas de entrenamiento
        """
        try:
            # Seleccionar features numéricas
            numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
            X = features_df[numeric_features].fillna(0)

            # Estandarizar features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Entrenar K-Means
            self.kmeans_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = self.kmeans_model.fit_predict(X_scaled)

            # Calcular métricas
            silhouette_avg = silhouette_score(X_scaled, labels)
            silhouette_values = silhouette_samples(X_scaled, labels)

            self.feature_names = numeric_features

            logger.info(f"Clustering trained successfully")
            logger.info(f"  - N Clusters: {self.n_clusters}")
            logger.info(f"  - Silhouette Score: {silhouette_avg:.4f}")
            logger.info(f"  - N Features: {len(numeric_features)}")

            # Estadísticas por cluster
            cluster_stats = {}
            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                cluster_size = mask.sum()
                cluster_silhouette = silhouette_values[mask].mean() if cluster_size > 0 else 0

                cluster_stats[cluster_id] = {
                    'name': self.cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
                    'size': int(cluster_size),
                    'percentage': float(cluster_size / len(labels) * 100),
                    'silhouette_score': float(cluster_silhouette),
                    'expected_risk': self.cluster_risk_mapping.get(cluster_id, 'unknown')
                }

            return {
                'success': True,
                'silhouette_score': float(silhouette_avg),
                'n_samples': len(X),
                'n_features': len(numeric_features),
                'cluster_stats': cluster_stats,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error training clustering: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def predict_cluster(self, features: Dict[str, float]) -> Dict:
        """
        Predecir cluster para un estudiante

        Args:
            features: Diccionario con features del estudiante

        Returns:
            Diccionario con predicción de cluster
        """
        try:
            if self.kmeans_model is None or self.scaler is None:
                raise ValueError("Model not trained. Train clustering first.")

            # Preparar features en el mismo orden que el entrenamiento
            feature_values = np.array([[features.get(fname, 0) for fname in self.feature_names]])

            # Estandarizar
            feature_scaled = self.scaler.transform(feature_values)

            # Predecir cluster
            cluster_id = int(self.kmeans_model.predict(feature_scaled)[0])

            # Calcular distancia a centroide (para probabilidad)
            distances = np.linalg.norm(
                feature_scaled - self.kmeans_model.cluster_centers_,
                axis=1
            )

            # Convertir distancia a probabilidad (cercano = alta probabilidad)
            min_distance = distances.min()
            probabilities = 1 / (1 + distances / (min_distance + 1e-10))
            probabilities = probabilities / probabilities.sum()
            cluster_probability = float(probabilities[cluster_id])

            # Distancia al centroide más cercano
            distance_to_centroid = float(min_distance)

            # Silhueta score (estimado)
            silhouette_estimated = float(2 * cluster_probability - 1)  # [-1, 1]

            return {
                'cluster_id': cluster_id,
                'cluster_name': self.cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
                'expected_risk': self.cluster_risk_mapping.get(cluster_id, 'unknown'),
                'cluster_probability': min(1.0, cluster_probability),
                'distance_to_centroid': distance_to_centroid,
                'silhouette_score': max(-1.0, min(1.0, silhouette_estimated)),
                'confidence': self._calculate_confidence(cluster_probability, distance_to_centroid),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error predicting cluster: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def batch_predict_clusters(self, features_list: List[Dict[str, float]]) -> Dict:
        """
        Predecir clusters para múltiples estudiantes

        Args:
            features_list: Lista de diccionarios con features

        Returns:
            Diccionario con predicciones por estudiante
        """
        try:
            if self.kmeans_model is None:
                raise ValueError("Model not trained")

            results = []
            for idx, features in enumerate(features_list):
                try:
                    prediction = self.predict_cluster(features)
                    results.append({
                        'index': idx,
                        'prediction': prediction,
                        'success': 'success' not in prediction
                    })
                except Exception as e:
                    results.append({
                        'index': idx,
                        'error': str(e),
                        'success': False
                    })

            return {
                'total': len(features_list),
                'successful': sum(1 for r in results if r['success']),
                'results': results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def save_model(self, filepath: str) -> bool:
        """
        Guardar modelo entrenado

        Args:
            filepath: Ruta donde guardar el modelo

        Returns:
            True si se guardó exitosamente
        """
        try:
            if self.kmeans_model is None:
                raise ValueError("No model trained to save")

            model_data = {
                'kmeans': self.kmeans_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'n_clusters': self.n_clusters,
                'cluster_names': self.cluster_names,
                'cluster_risk_mapping': self.cluster_risk_mapping,
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Clustering model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving clustering model: {str(e)}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Cargar modelo entrenado

        Args:
            filepath: Ruta del modelo guardado

        Returns:
            True si se cargó exitosamente
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.kmeans_model = model_data['kmeans']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.n_clusters = model_data['n_clusters']
            self.cluster_names = model_data.get('cluster_names', self.cluster_names)
            self.cluster_risk_mapping = model_data.get('cluster_risk_mapping', self.cluster_risk_mapping)

            logger.info(f"Clustering model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading clustering model: {str(e)}")
            return False

    def _calculate_confidence(self, probability: float, distance: float) -> float:
        """
        Calcular confianza basada en probabilidad y distancia

        Args:
            probability: Probabilidad del cluster predicho
            distance: Distancia al centroide

        Returns:
            Confianza entre 0 y 1
        """
        # Probabilidad contribuye 70%
        prob_component = probability * 0.7

        # Distancia normalizada contribuye 30%
        # Cercano = alta confianza, lejano = baja confianza
        distance_component = max(0, 1 - min(1, distance / 10)) * 0.3

        return min(1.0, prob_component + distance_component)

    def get_cluster_info(self, cluster_id: int) -> Dict:
        """
        Obtener información sobre un cluster

        Args:
            cluster_id: ID del cluster

        Returns:
            Información del cluster
        """
        if cluster_id not in self.cluster_names:
            return {'error': f'Invalid cluster_id: {cluster_id}'}

        return {
            'cluster_id': cluster_id,
            'cluster_name': self.cluster_names[cluster_id],
            'expected_risk': self.cluster_risk_mapping[cluster_id],
            'description': self._get_cluster_description(cluster_id)
        }

    def _get_cluster_description(self, cluster_id: int) -> str:
        """Obtener descripción del cluster"""
        descriptions = {
            0: "Estudiantes con bajo desempeño académico. Requieren atención y apoyo inmediato.",
            1: "Estudiantes con desempeño medio. Tienen potencial pero necesitan seguimiento.",
            2: "Estudiantes con alto desempeño. Mantienen consistencia académica positiva.",
        }
        return descriptions.get(cluster_id, "Cluster sin descripción")
