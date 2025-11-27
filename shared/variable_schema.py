"""
Variable Schema Unificado para ML Pipeline
Plataforma Educativa

Define nombres de variables consistentes en todo el pipeline Python-PHP-BD
"""

# ============================================================================
# SCHEMA UNIFICADO - Nombres consistentes entre capas
# ============================================================================

UNIFIED_SCHEMA = {
    # Identificadores y timestamps
    'prediction_id': {'type': 'int', 'db_column': 'id'},
    'student_id': {'type': 'int', 'db_column': 'student_id'},
    'model_name': {'type': 'str', 'db_column': 'model_name'},
    'model_version': {'type': 'str', 'db_column': 'model_version'},
    'prediction_timestamp': {'type': 'datetime', 'db_column': 'created_at'},

    # RISK PREDICTION - Nombres Unificados
    'risk_score': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Score de riesgo de bajo desempeño (0=sin riesgo, 1=máximo riesgo)',
        'db_column': 'score_riesgo',
        'php_var': 'score_riesgo',
        'python_var': 'risk_score',
        'source': 'predict_proba()[0]'
    },
    'risk_level': {
        'type': 'enum',
        'values': ['alto', 'medio', 'bajo'],
        'description': 'Nivel de riesgo categorizado',
        'db_column': 'nivel_riesgo',
        'php_var': 'nivel_riesgo',
        'python_var': 'risk_level',
        'thresholds': {
            'alto': (0.70, 1.0),
            'medio': (0.40, 0.70),
            'bajo': (0.0, 0.40)
        }
    },
    'risk_confidence': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Confianza del modelo en la predicción de riesgo',
        'db_column': 'confianza',
        'php_var': 'confianza',
        'python_var': 'risk_confidence',
        'source': 'max(predict_proba()[i] for i in [0,1])'
    },

    # CAREER RECOMMENDATION
    'career_name': {
        'type': 'str',
        'description': 'Nombre de carrera recomendada',
        'db_column': 'carrera_nombre',
        'php_var': 'carrera_nombre',
        'python_var': 'career_name'
    },
    'career_compatibility': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Compatibilidad del estudiante con la carrera',
        'db_column': 'compatibilidad',
        'php_var': 'compatibilidad',
        'python_var': 'compatibility'
    },
    'career_ranking': {
        'type': 'int',
        'range': [1, 3],
        'description': 'Ranking de recomendación (1=más recomendada)',
        'db_column': 'ranking',
        'php_var': 'ranking',
        'python_var': 'ranking'
    },
    'career_confidence': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Confianza en la recomendación de carrera',
        'db_column': 'confianza',
        'php_var': 'confianza',
        'python_var': 'career_confidence',
        'source': 'max(predict_proba())'
    },

    # TREND PREDICTION
    'trend_type': {
        'type': 'enum',
        'values': ['mejorando', 'estable', 'declinando', 'fluctuando'],
        'description': 'Tendencia de desempeño del estudiante',
        'db_column': 'tendencia',
        'php_var': 'tendencia',
        'python_var': 'trend'
    },
    'trend_confidence': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Confianza en la predicción de tendencia',
        'db_column': 'confianza',
        'php_var': 'confianza',
        'python_var': 'trend_confidence',
        'source': 'max(predict_proba())'
    },

    # PROGRESS ANALYSIS
    'learning_velocity': {
        'type': 'float',
        'description': 'Velocidad de aprendizaje (cambio de promedio por período)',
        'db_column': 'velocidad_aprendizaje',
        'php_var': 'velocidad_aprendizaje',
        'python_var': 'learning_velocity'
    },
    'projected_grade': {
        'type': 'float',
        'range': [0, 100],
        'description': 'Calificación proyectada para el período',
        'db_column': 'nota_proyectada',
        'php_var': 'nota_proyectada',
        'python_var': 'projected_grade'
    },
    'progress_confidence': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Confianza en la predicción de progreso',
        'db_column': 'confianza',
        'php_var': 'confianza',
        'python_var': 'progress_confidence'
    },

    # CLUSTERING
    'cluster_id': {
        'type': 'int',
        'range': [0, 2],
        'description': 'ID del cluster de K-Means asignado',
        'values': {
            0: 'Bajo Desempeño',
            1: 'Desempeño Medio',
            2: 'Alto Desempeño'
        },
        'db_column': 'cluster_id',
        'php_var': 'cluster_id',
        'python_var': 'cluster_id',
        'source': 'kmeans.labels_[student_idx]'
    },
    'cluster_probability': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Probabilidad de pertenencia al cluster',
        'db_column': 'cluster_probability',
        'php_var': 'cluster_probability',
        'python_var': 'cluster_probability'
    },
    'cluster_silhouette': {
        'type': 'float',
        'range': [-1, 1],
        'description': 'Score de silhueta para calidad del cluster',
        'db_column': 'silhouette_score',
        'php_var': 'silhouette_score',
        'python_var': 'silhouette_score'
    },

    # ANOMALY DETECTION
    'anomaly_score': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Score de anomalía detectado por LSTM',
        'db_column': 'anomaly_score',
        'php_var': 'anomaly_score',
        'python_var': 'anomaly_score'
    },
    'anomaly_type': {
        'type': 'enum',
        'values': ['spike_down', 'spike_up', 'volatility', 'drift', None],
        'description': 'Tipo de anomalía detectada',
        'db_column': 'anomaly_type',
        'php_var': 'anomaly_type',
        'python_var': 'anomaly_type'
    },

    # MODEL QUALITY METRICS
    'feature_importance': {
        'type': 'json',
        'description': 'Importancia de features usados en la predicción',
        'db_column': 'feature_importance',
        'php_var': 'caracteristicas_importancia',
        'python_var': 'feature_importance'
    },
    'model_precision': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Precisión del modelo (entrenamiento/validación)',
        'db_column': 'model_precision',
        'php_var': 'model_precision',
        'python_var': 'precision'
    },
    'model_recall': {
        'type': 'float',
        'range': [0, 1],
        'description': 'Recall del modelo',
        'db_column': 'model_recall',
        'php_var': 'model_recall',
        'python_var': 'recall'
    },
    'model_f1_score': {
        'type': 'float',
        'range': [0, 1],
        'description': 'F1-score del modelo',
        'db_column': 'model_f1_score',
        'php_var': 'model_f1_score',
        'python_var': 'f1_score'
    },
}

# ============================================================================
# THRESHOLDS Y LÍMITES
# ============================================================================

THRESHOLDS = {
    'risk': {
        'high': 0.70,
        'medium_low': 0.40,
        'low': 0.0,
    },
    'anomaly': {
        'threshold': 0.8,  # Si anomaly_score >= 0.8 → es anomalía
    },
    'coherence': {
        'tolerance': 0.2,  # Tolerancia para detectar inconsistencias
    },
    'confidence': {
        'high': 0.85,
        'medium': 0.60,
        'low': 0.0,
    },
    'silhouette': {
        'good': 0.5,
        'fair': 0.25,
        'poor': 0.0,
    }
}

# ============================================================================
# VALIDACIÓN DE COHERENCIA
# ============================================================================

COHERENCE_RULES = {
    'risk_career_contradiction': {
        'rule': 'IF risk_score >= 0.70 AND career_compatibility > 0.75',
        'severity': 'warning',
        'action': 'verify_data_quality',
        'explanation': 'Estudiante con alto riesgo no debería tener carrera altamente compatible'
    },
    'trend_risk_contradiction': {
        'rule': 'IF trend == "mejorando" AND risk_level == "alto"',
        'severity': 'info',
        'action': 'monitor',
        'explanation': 'Tendencia mejorando pero riesgo alto (posible lag de datos)'
    },
    'cluster_risk_mismatch': {
        'rule': 'expected_risk_by_cluster[cluster_id] != risk_level',
        'severity': 'warning',
        'action': 'investigate',
        'explanation': 'Cluster assignment no coincide con predicción de riesgo'
    },
    'lstm_anomaly_not_flagged': {
        'rule': 'anomaly_score >= THRESHOLDS["anomaly"] AND risk_level != "alto"',
        'severity': 'warning',
        'action': 'escalate_risk',
        'explanation': 'LSTM detectó anomalía pero riesgo no fue escalado'
    }
}

# ============================================================================
# MAPPINGS ENTRE CAPAS (Python ↔ PHP ↔ BD)
# ============================================================================

class VariableMapper:
    """Facilita conversión de nombres entre capas"""

    @staticmethod
    def python_to_php(python_var_name: str) -> str:
        """Convertir nombre de variable Python → PHP"""
        for unified_key, schema in UNIFIED_SCHEMA.items():
            if isinstance(schema, dict) and schema.get('python_var') == python_var_name:
                return schema.get('php_var', python_var_name)
        return python_var_name

    @staticmethod
    def python_to_db(python_var_name: str) -> str:
        """Convertir nombre de variable Python → BD"""
        for unified_key, schema in UNIFIED_SCHEMA.items():
            if isinstance(schema, dict) and schema.get('python_var') == python_var_name:
                return schema.get('db_column', python_var_name)
        return python_var_name

    @staticmethod
    def php_to_db(php_var_name: str) -> str:
        """Convertir nombre de variable PHP → BD"""
        for unified_key, schema in UNIFIED_SCHEMA.items():
            if isinstance(schema, dict) and schema.get('php_var') == php_var_name:
                return schema.get('db_column', php_var_name)
        return php_var_name

    @staticmethod
    def standardize_prediction(prediction_dict: dict, source: str = 'python') -> dict:
        """
        Estandarizar predicción usando nombres unificados

        Args:
            prediction_dict: Dictionary con predicción cruda
            source: 'python' o 'php'

        Returns:
            Dictionary con nombres estandarizados
        """
        standardized = {}

        # Buscar match para cada clave
        for key, value in prediction_dict.items():
            found = False
            for unified_key, schema in UNIFIED_SCHEMA.items():
                if isinstance(schema, dict):
                    source_field = schema.get(f'{source}_var')
                    if source_field == key:
                        standardized[unified_key] = value
                        found = True
                        break

            if not found:
                # Conservar key original si no hay mapping
                standardized[key] = value

        return standardized


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo 1: Convertir nombres
    print("=== CONVERSIÓN DE NOMBRES ===")
    print(f"Python 'risk_score' → PHP: {VariableMapper.python_to_php('risk_score')}")
    print(f"Python 'risk_score' → BD: {VariableMapper.python_to_db('risk_score')}")

    # Ejemplo 2: Estandarizar predicción desde Python
    prediction_from_python = {
        'risk_score': 0.75,
        'risk_level': 'alto',
        'risk_confidence': 0.92,
        'career_name': 'Ingeniería',
        'compatibility': 0.65,
    }

    standardized = VariableMapper.standardize_prediction(
        prediction_from_python,
        source='python'
    )
    print("\n=== PREDICCIÓN ESTANDARIZADA ===")
    for key, value in standardized.items():
        print(f"{key}: {value}")
