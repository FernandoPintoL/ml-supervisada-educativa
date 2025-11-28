"""
Script para regenerar caché de dataset
Mejora performance al tener datos precargados
"""

import sys
import logging
from pathlib import Path

# Agregar directorio actual al path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from cache.cache_manager import CacheManager
from data.data_processor import DataProcessor
from shared.config import DEBUG, LOG_LEVEL

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def regenerate_cache():
    """Regenerar caché del dataset"""
    try:
        logger.info("=" * 60)
        logger.info("REGENERANDO CACHE DE DATASET")
        logger.info("=" * 60)
        
        cache_manager = CacheManager()
        
        # Feature names que espera el modelo
        feature_names = [
            'promedio_calificaciones',
            'varianza_calificaciones',
            'max_calificacion',
            'min_calificacion',
            'num_calificaciones',
            'num_trabajos',
            'promedio_intentos',
            'dias_promedio_entrega',
            'promedio_consultas_material',
            'trabajos_entregados',
            'trabajos_calificados'
        ]
        
        logger.info(f"Features esperados ({len(feature_names)}): {feature_names}")
        
        # Intentar cargar datos desde BD
        logger.info("Cargando datos desde base de datos...")
        try:
            # Este es un placeholder - en producción necesitarías
            # conectar a tu BD real
            logger.warning("Nota: Se necesita conexión a BD para cargar datos reales")
            logger.warning("Para ahora, creando dataset de prueba...")
            
            import numpy as np
            import pandas as pd
            
            # Crear datos de prueba
            n_samples = 100
            data = {
                'student_id': np.arange(1, n_samples + 1),
                'promedio_calificaciones': np.random.uniform(0, 100, n_samples),
                'varianza_calificaciones': np.random.uniform(0, 50, n_samples),
                'max_calificacion': np.random.uniform(70, 100, n_samples),
                'min_calificacion': np.random.uniform(0, 70, n_samples),
                'num_calificaciones': np.random.randint(5, 50, n_samples),
                'num_trabajos': np.random.randint(1, 20, n_samples),
                'promedio_intentos': np.random.uniform(1, 5, n_samples),
                'dias_promedio_entrega': np.random.uniform(0, 30, n_samples),
                'promedio_consultas_material': np.random.uniform(0, 100, n_samples),
                'trabajos_entregados': np.random.randint(0, 20, n_samples),
                'trabajos_calificados': np.random.randint(0, 20, n_samples),
            }
            
            df = pd.DataFrame(data)
            logger.info(f"Dataset de prueba creado: {df.shape[0]} registros, {df.shape[1]} caracteristicas")
            
            # Metadata
            metadata = {
                'source': 'test_data',
                'description': 'Dataset de prueba para cache',
                'total_records': len(df),
                'features_count': len(feature_names)
            }
            
            # Guardar en caché
            logger.info("Guardando en cache...")
            success = cache_manager.save_dataset(df, feature_names, metadata)
            
            if success:
                # Verificar caché
                info = cache_manager.get_cache_info()
                logger.info("Informacion del cache:")
                logger.info(f"  - Existe: {info['exists']}")
                logger.info(f"  - Registros: {info['num_records']}")
                logger.info(f"  - Features: {info['num_features']}")
                logger.info(f"  - Tamano: {info['size_mb']}MB")
                logger.info(f"  - Timestamp: {info['timestamp']}")
                
                logger.info("=" * 60)
                logger.info("CACHE REGENERADO EXITOSAMENTE")
                logger.info("=" * 60)
                return True
            else:
                logger.error("No se pudo guardar el cache")
                return False
            
        except Exception as e:
            logger.error(f"Error al regenerar cache: {str(e)}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"Error general: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    success = regenerate_cache()
    sys.exit(0 if success else 1)
