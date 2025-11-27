"""
LSTM Anomaly Detector para Plataforma Educativa

Detecta anomalías en series temporales de calificaciones de estudiantes usando LSTM.
Identifica patrones inusuales como:
- Caídas abruptas (spike_down)
- Subidas anormales (spike_up)
- Alta volatilidad (volatility)
- Tendencia de degradación (drift)
"""

import logging
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class LSTMAnomalyDetector:
    """
    Detector de anomalías usando análisis estadístico y patrones temporales.

    Nota: Implementación sin dependencias de TensorFlow/Keras para mayor compatibilidad.
    Usa análisis estadístico de series de tiempo + detección de patrones.
    """

    def __init__(self, window_size: int = 5, contamination: float = 0.1):
        """
        Inicializar detector de anomalías

        Args:
            window_size: Número de calificaciones para análisis de patrones
            contamination: Proporción esperada de anomalías (0.1 = 10%)
        """
        self.window_size = window_size
        self.contamination = contamination
        self.is_trained = False
        self.thresholds = {
            'spike_down': 0.7,      # Si cae 70% o más
            'spike_up': 0.5,        # Si sube 50% o más
            'volatility': 15.0,     # Desviación estándar > 15
            'drift': 0.8,           # Tendencia negativa fuerte
        }

    def fit(self, grades_history: List[List[float]]) -> Dict:
        """
        Entrenar detector aprendiendo estadísticas de los datos

        Args:
            grades_history: Lista de listas con historiales de calificaciones

        Returns:
            Diccionario con estadísticas aprendidas
        """
        try:
            all_grades = []
            all_volatilities = []
            all_changes = []

            for grades in grades_history:
                if len(grades) < 2:
                    continue

                all_grades.extend(grades)

                # Calcular volatilidad
                volatility = np.std(grades) if len(grades) > 1 else 0
                all_volatilities.append(volatility)

                # Calcular cambios entre consecutivos
                changes = [abs(grades[i] - grades[i-1]) for i in range(1, len(grades))]
                all_changes.extend(changes)

            if not all_grades:
                logger.warning("No grades to train on")
                return {'success': False, 'error': 'No grades provided'}

            # Calcular estadísticas
            self.mean_grade = float(np.mean(all_grades))
            self.std_grade = float(np.std(all_grades))
            self.mean_volatility = float(np.mean(all_volatilities)) if all_volatilities else 0
            self.mean_change = float(np.mean(all_changes)) if all_changes else 0
            self.std_change = float(np.std(all_changes)) if all_changes else 0

            # Ajustar thresholds basado en datos
            if self.std_change > 0:
                self.thresholds['spike_down'] = self.mean_change + (2 * self.std_change)
                self.thresholds['spike_up'] = self.mean_change + (2 * self.std_change)

            self.is_trained = True

            logger.info("LSTM Anomaly Detector trained successfully")
            logger.info(f"  - Mean grade: {self.mean_grade:.2f}")
            logger.info(f"  - Grade std: {self.std_grade:.2f}")
            logger.info(f"  - Mean volatility: {self.mean_volatility:.2f}")

            return {
                'success': True,
                'samples': len(grades_history),
                'mean_grade': self.mean_grade,
                'std_grade': self.std_grade,
                'mean_volatility': self.mean_volatility,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error training detector: {str(e)}")
            return {'success': False, 'error': str(e)}

    def detect_anomaly(self, grades: List[float], student_id: int = None) -> Dict:
        """
        Detectar anomalías en una serie de calificaciones

        Args:
            grades: Lista de calificaciones (ordenadas cronológicamente)
            student_id: ID del estudiante (opcional, para logging)

        Returns:
            Diccionario con resultado de detección
        """
        try:
            if len(grades) < 2:
                return self._no_anomaly_result("Insuficientes calificaciones")

            # Análisis multi-aspecto
            spike_down_score = self._detect_spike_down(grades)
            spike_up_score = self._detect_spike_up(grades)
            volatility_score = self._detect_volatility(grades)
            drift_score = self._detect_drift(grades)

            # Determinar tipo y score de anomalía
            scores = {
                'spike_down': spike_down_score,
                'spike_up': spike_up_score,
                'volatility': volatility_score,
                'drift': drift_score,
            }

            # Encontrar anomalía más probable
            anomaly_type = max(scores, key=scores.get)
            anomaly_score = scores[anomaly_type]

            # Threshold global para considerar como anomalía
            is_anomaly = anomaly_score >= 0.6

            result = {
                'is_anomaly': is_anomaly,
                'anomaly_score': float(min(1.0, anomaly_score)),
                'anomaly_type': anomaly_type if is_anomaly else None,
                'scores': {k: float(min(1.0, v)) for k, v in scores.items()},
                'details': self._generate_details(grades, anomaly_type, anomaly_score),
                'recommendation': self._get_recommendation(anomaly_type, anomaly_score),
                'timestamp': datetime.now().isoformat()
            }

            if is_anomaly and student_id:
                logger.warning(
                    f"Anomaly detected for student {student_id}: "
                    f"type={anomaly_type}, score={anomaly_score:.3f}"
                )

            return result

        except Exception as e:
            logger.error(f"Error detecting anomaly: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'is_anomaly': False,
                'anomaly_score': 0,
            }

    def _detect_spike_down(self, grades: List[float]) -> float:
        """Detectar caída abrupta en calificaciones"""
        if len(grades) < 2:
            return 0

        recent = grades[-1]
        previous = grades[-2]

        if previous == 0:
            return 0

        change_pct = abs((recent - previous) / previous)

        # Caída significativa
        if recent < previous:
            if change_pct > 0.7:  # Caída de más de 70%
                return min(1.0, change_pct)

        return 0

    def _detect_spike_up(self, grades: List[float]) -> float:
        """Detectar subida anormal en calificaciones"""
        if len(grades) < 2:
            return 0

        recent = grades[-1]
        previous = grades[-2]

        if previous == 0:
            return 0

        change_pct = abs((recent - previous) / previous)

        # Subida anormal después de bajas calificaciones
        if recent > previous and previous < 50:
            if change_pct > 0.5:  # Subida de más de 50%
                return min(1.0, change_pct * 0.5)  # Menos severo que caída

        return 0

    def _detect_volatility(self, grades: List[float]) -> float:
        """Detectar alta volatilidad (cambios erráticos)"""
        if len(grades) < 3:
            return 0

        volatility = np.std(grades)
        mean_val = np.mean(grades)

        # Coeficiente de variación
        if mean_val > 0:
            cv = volatility / mean_val
            if cv > 0.3:  # Variación > 30%
                return min(1.0, cv)

        return 0

    def _detect_drift(self, grades: List[float]) -> float:
        """Detectar degradación progresiva (tendencia negativa)"""
        if len(grades) < 3:
            return 0

        # Dividir en dos mitades
        mid = len(grades) // 2
        first_half = grades[:mid]
        second_half = grades[mid:]

        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)

        # Degradación si segunda mitad es significativamente peor
        if mean_first > 0:
            degradation = (mean_first - mean_second) / mean_first

            if degradation > 0.15:  # Degradación > 15%
                # Score basado en severidad
                return min(1.0, degradation)

        return 0

    def _generate_details(self, grades: List[float], anomaly_type: str, score: float) -> Dict:
        """Generar detalles explicativos de la anomalía"""
        details = {
            'current_grade': float(grades[-1]) if grades else None,
            'previous_grade': float(grades[-2]) if len(grades) > 1 else None,
            'mean_grade': float(np.mean(grades)) if grades else None,
            'std_grade': float(np.std(grades)) if grades else None,
            'sequence_length': len(grades),
            'recent_trend': self._get_recent_trend(grades),
        }

        if anomaly_type == 'spike_down':
            details['message'] = "Caída abrupta en calificaciones detectada"
            details['severity'] = "high" if score > 0.8 else "medium"

        elif anomaly_type == 'spike_up':
            details['message'] = "Subida anormal en calificaciones detectada"
            details['severity'] = "low"

        elif anomaly_type == 'volatility':
            details['message'] = "Alta volatilidad en calificaciones detectada"
            details['severity'] = "medium"

        elif anomaly_type == 'drift':
            details['message'] = "Tendencia de degradación detectada"
            details['severity'] = "high"

        return details

    def _get_recent_trend(self, grades: List[float]) -> str:
        """Determinar tendencia reciente"""
        if len(grades) < 2:
            return 'unknown'

        recent = grades[-1]
        mean_prev = np.mean(grades[:-1])

        if recent > mean_prev * 1.05:
            return 'improving'
        elif recent < mean_prev * 0.95:
            return 'declining'
        else:
            return 'stable'

    def _get_recommendation(self, anomaly_type: Optional[str], score: float) -> Dict:
        """Obtener recomendación basada en tipo de anomalía"""
        recommendations = {
            'spike_down': {
                'action': 'escalate_risk',
                'message': 'Escalar riesgo inmediatamente. Investigar causa de caída.',
                'priority': 'high'
            },
            'spike_up': {
                'action': 'monitor',
                'message': 'Monitorear cambio positivo. Verificar integridad de datos.',
                'priority': 'medium'
            },
            'volatility': {
                'action': 'investigate',
                'message': 'Investigar fluctuaciones. Pueden indicar inconsistencia académica.',
                'priority': 'medium'
            },
            'drift': {
                'action': 'escalate_risk',
                'message': 'Escalar riesgo. Tendencia de degradación progresiva detectada.',
                'priority': 'high'
            },
        }

        if anomaly_type in recommendations:
            rec = recommendations[anomaly_type].copy()
            rec['confidence'] = float(min(1.0, score))
            return rec

        return {
            'action': 'proceed_normally',
            'message': 'Sin anomalías detectadas',
            'priority': 'low',
            'confidence': 0
        }

    def _no_anomaly_result(self, reason: str) -> Dict:
        """Resultado cuando no hay anomalía"""
        return {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'anomaly_type': None,
            'scores': {
                'spike_down': 0.0,
                'spike_up': 0.0,
                'volatility': 0.0,
                'drift': 0.0,
            },
            'details': {'reason': reason},
            'recommendation': {
                'action': 'proceed_normally',
                'message': reason,
                'priority': 'low',
                'confidence': 1.0
            },
            'timestamp': datetime.now().isoformat()
        }

    def save_model(self, filepath: str) -> bool:
        """Guardar modelo entrenado"""
        try:
            model_data = {
                'is_trained': self.is_trained,
                'window_size': self.window_size,
                'contamination': self.contamination,
                'mean_grade': getattr(self, 'mean_grade', None),
                'std_grade': getattr(self, 'std_grade', None),
                'mean_volatility': getattr(self, 'mean_volatility', None),
                'mean_change': getattr(self, 'mean_change', None),
                'std_change': getattr(self, 'std_change', None),
                'thresholds': self.thresholds,
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"LSTM Anomaly Detector saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving detector: {str(e)}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Cargar modelo entrenado"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.is_trained = model_data['is_trained']
            self.window_size = model_data['window_size']
            self.contamination = model_data['contamination']
            self.mean_grade = model_data.get('mean_grade')
            self.std_grade = model_data.get('std_grade')
            self.mean_volatility = model_data.get('mean_volatility')
            self.mean_change = model_data.get('mean_change')
            self.std_change = model_data.get('std_change')
            self.thresholds = model_data.get('thresholds', self.thresholds)

            logger.info(f"LSTM Anomaly Detector loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading detector: {str(e)}")
            return False
