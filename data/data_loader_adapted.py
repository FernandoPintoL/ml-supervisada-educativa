"""
Data Loader Adaptado para Plataforma Educativa Real
Plataforma Educativa ML

Carga datos desde la estructura real de PostgreSQL:
- Calificaciones: tabla calificaciones
- Trabajos: tabla trabajos
- Rendimiento: tabla rendimiento_academico
- Cursos: tabla cursos con relación curso_estudiante
"""

import logging
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from shared.database.connection import get_db_session, DBSession
from shared.config import (
    DATABASE_URL,
    MIN_STUDENTS_REQUIRED,
    DEBUG
)

# Configurar logger
logger = logging.getLogger(__name__)


class DataLoaderAdapted:
    """
    Cargador de datos desde PostgreSQL usando estructura real.
    Adapta a las tablas: calificaciones, trabajos, cursos, etc.
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Inicializar data loader.

        Args:
            db_session (Session): Sesión SQLAlchemy (crea una nueva si no se proporciona)
        """
        self.db = db_session
        self.own_session = False

        if self.db is None:
            self.db = get_db_session()
            self.own_session = True

        if self.db is None:
            logger.error("No se pudo establecer conexión a la base de datos")

        logger.info("✓ DataLoaderAdapted inicializado")

    def _get_students_from_db(self) -> List[int]:
        """Obtener IDs de estudiantes existentes desde BD."""
        try:
            # Estrategia 1: Spatie Permissions
            logger.info("Obteniendo estudiantes con Spatie...")
            query = """
                SELECT DISTINCT u.id
                FROM users u
                INNER JOIN model_has_roles mhr ON u.id = mhr.model_id
                INNER JOIN roles r ON mhr.role_id = r.id
                WHERE mhr.model_type = 'App\\\\Models\\\\User'
                  AND r.name = 'student'
            """
            try:
                result = self.db.execute(text(query))
                self.db.commit()
                student_ids = [row[0] for row in result.fetchall()]
                if student_ids:
                    logger.info(f"✓ Encontrados {len(student_ids)} estudiantes")
                    return student_ids
            except Exception as e:
                logger.debug(f"Estrategia Spatie falló: {str(e)}")
                self.db.rollback()

            # Fallback: usuarios que tienen trabajos
            logger.info("Fallback: buscando estudiantes con trabajos...")
            query = "SELECT DISTINCT estudiante_id FROM trabajos LIMIT 50"
            try:
                result = self.db.execute(text(query))
                self.db.commit()
                student_ids = [row[0] for row in result.fetchall()]
                if student_ids:
                    logger.info(f"✓ Encontrados {len(student_ids)} estudiantes con trabajos")
                    return student_ids
            except:
                self.db.rollback()

            # Fallback final: primeros usuarios
            logger.warning("⚠ Fallback: usando primeros 10 usuarios")
            query = "SELECT id FROM users LIMIT 10"
            result = self.db.execute(text(query))
            self.db.commit()
            return [row[0] for row in result.fetchall()]

        except Exception as e:
            logger.error(f"✗ Error obteniendo estudiantes: {str(e)}", exc_info=True)
            try:
                self.db.rollback()
            except:
                pass
            return []

    def load_students(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Cargar datos básicos de estudiantes.

        Retorna:
            DataFrame: Estudiantes [id, name, email, role, created_at]
        """
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return pd.DataFrame()

            query = """
                SELECT
                    u.id,
                    u.name,
                    u.email,
                    'student' as role,
                    u.created_at
                FROM users u
                WHERE u.tipo_usuario = 'estudiante'
            """

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql(query, self.db.bind)
            if df.empty:
                logger.warning("⚠ No se encontraron estudiantes, usando primeros 10 usuarios")
                query = "SELECT id, name, email, 'student' as role, created_at FROM users LIMIT 10"
                df = pd.read_sql(query, self.db.bind)

            logger.info(f"✓ Cargados {len(df)} estudiantes")
            return df

        except Exception as e:
            logger.error(f"✗ Error cargando estudiantes: {str(e)}")
            return pd.DataFrame()

    def load_grades(self, student_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Cargar calificaciones desde tabla calificaciones.

        Usa estructura real: calificaciones -> trabajos -> estudiante_id

        Retorna:
            DataFrame: [estudiante_id, puntaje, fecha_calificacion]
        """
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return pd.DataFrame()

            query = """
                SELECT
                    t.estudiante_id as student_id,
                    c.puntaje as grade,
                    c.fecha_calificacion as evaluated_at
                FROM calificaciones c
                INNER JOIN trabajos t ON c.trabajo_id = t.id
                WHERE t.estudiante_id IS NOT NULL
            """

            if student_ids:
                placeholders = ','.join(str(id) for id in student_ids)
                query += f" AND t.estudiante_id IN ({placeholders})"

            query += " ORDER BY c.fecha_calificacion DESC"

            df = pd.read_sql(query, self.db.bind)
            logger.info(f"✓ Cargadas {len(df)} calificaciones")
            return df

        except Exception as e:
            logger.error(f"✗ Error cargando calificaciones: {str(e)}")
            return pd.DataFrame()

    def load_work_submissions(self, student_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Cargar datos de entregas de trabajos.

        Características:
        - Número de intentos
        - Tiempo de respuesta (fecha_entrega - fecha_inicio)
        - Consultas de material
        - Estado de entrega

        Retorna:
            DataFrame: [estudiante_id, intentos, tiempo_respuesta_dias, consultas_material]
        """
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return pd.DataFrame()

            query = """
                SELECT
                    estudiante_id as student_id,
                    COUNT(*) as num_trabajos,
                    AVG(intentos) as promedio_intentos,
                    AVG(EXTRACT(DAY FROM (fecha_entrega - fecha_inicio))) as dias_promedio_entrega,
                    AVG(consultas_material) as promedio_consultas_material,
                    SUM(CASE WHEN estado = 'entregado' THEN 1 ELSE 0 END) as trabajos_entregados,
                    SUM(CASE WHEN estado = 'calificado' THEN 1 ELSE 0 END) as trabajos_calificados
                FROM trabajos
                WHERE estudiante_id IS NOT NULL
                  AND fecha_entrega IS NOT NULL
            """

            if student_ids:
                placeholders = ','.join(str(id) for id in student_ids)
                query += f" AND estudiante_id IN ({placeholders})"

            query += " GROUP BY estudiante_id"

            df = pd.read_sql(query, self.db.bind)
            logger.info(f"✓ Cargados datos de {len(df)} estudiantes")
            return df

        except Exception as e:
            logger.error(f"✗ Error cargando datos de trabajos: {str(e)}")
            return pd.DataFrame()

    def load_academic_performance(self, student_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Cargar rendimiento académico histórico.

        Retorna:
            DataFrame: [estudiante_id, promedio, tendencia_temporal]
        """
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return pd.DataFrame()

            query = """
                SELECT
                    estudiante_id as student_id,
                    promedio,
                    tendencia_temporal,
                    created_at
                FROM rendimiento_academico
                WHERE estudiante_id IS NOT NULL
            """

            if student_ids:
                placeholders = ','.join(str(id) for id in student_ids)
                query += f" AND estudiante_id IN ({placeholders})"

            query += " ORDER BY created_at DESC"

            df = pd.read_sql(query, self.db.bind)

            # Agregar por estudiante (usar el registro más reciente)
            if not df.empty:
                df = df.drop_duplicates(subset=['student_id'], keep='first')

            logger.info(f"✓ Cargado rendimiento de {len(df)} estudiantes")
            return df

        except Exception as e:
            logger.error(f"✗ Error cargando rendimiento académico: {str(e)}")
            return pd.DataFrame()

    def aggregate_student_features(self, df_grades: pd.DataFrame,
                                   df_works: pd.DataFrame) -> pd.DataFrame:
        """
        Agregar features de estudiante desde calificaciones y trabajos.

        Features creadas:
        - promedio_calificaciones
        - varianza_calificaciones
        - promedio_intentos
        - dias_promedio_entrega
        - consultas_material_promedio
        - tasa_entrega

        Retorna:
            DataFrame: Features agregadas por estudiante
        """
        try:
            if df_grades.empty and df_works.empty:
                logger.warning("DataFrames vacíos para agregación")
                return pd.DataFrame()

            # Agregar calificaciones
            if not df_grades.empty:
                grade_stats = df_grades.groupby('student_id')['grade'].agg([
                    ('promedio_calificaciones', 'mean'),
                    ('varianza_calificaciones', 'var'),
                    ('max_calificacion', 'max'),
                    ('min_calificacion', 'min'),
                    ('num_calificaciones', 'count')
                ]).reset_index()
            else:
                grade_stats = pd.DataFrame()

            # Works ya viene agregado
            works_stats = df_works.copy() if not df_works.empty else pd.DataFrame()

            # Unir
            if not grade_stats.empty and not works_stats.empty:
                result = grade_stats.merge(works_stats, on='student_id', how='outer')
            elif not grade_stats.empty:
                result = grade_stats
            else:
                result = works_stats

            # Llenar NaN
            result = result.fillna(0)

            logger.info(f"✓ Features agregadas para {len(result)} estudiantes")
            return result

        except Exception as e:
            logger.error(f"✗ Error agregando features: {str(e)}")
            return pd.DataFrame()

    def load_training_data(self, limit: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Cargar y preparar datos completos para entrenamiento.

        Retorna:
            Tuple[DataFrame, List[str]]: Datos listos para ML + nombres de features
        """
        try:
            logger.info("Iniciando carga de datos de entrenamiento...")

            # 1. Cargar estudiantes
            students = self.load_students(limit=limit)
            if len(students) < MIN_STUDENTS_REQUIRED:
                logger.warning(
                    f"Menos de {MIN_STUDENTS_REQUIRED} estudiantes disponibles. "
                    f"Actual: {len(students)}"
                )

            student_ids = students['id'].tolist()

            # 2. Cargar datos
            grades = self.load_grades(student_ids=student_ids)
            works = self.load_work_submissions(student_ids=student_ids)

            # 3. Agregar features
            stats = self.aggregate_student_features(grades, works)

            # 4. Unir
            if not stats.empty:
                result = students.merge(stats, left_on='id', right_on='student_id', how='inner')  # inner join para mantener sincronía
                result = result.fillna(0)
                logger.info(f"✓ Datos preparados: {result.shape}")
            else:
                result = students
                logger.warning("No se pudieron agregar estadísticas")

            # Features disponibles
            features = [col for col in result.columns if col not in ['id', 'name', 'email', 'role', 'created_at', 'student_id']]

            return result, features

        except Exception as e:
            logger.error(f"✗ Error cargando datos: {str(e)}")
            return pd.DataFrame(), []

    def test_connection(self) -> bool:
        """Probar conexión."""
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return False

            self.db.execute(text("SELECT 1"))
            logger.info("✓ Conexión a BD verificada")
            return True

        except Exception as e:
            logger.error(f"✗ Error testando conexión: {str(e)}")
            return False

    def close(self) -> None:
        """Cerrar sesión."""
        if self.own_session and self.db:
            self.db.close()
            logger.info("DataLoaderAdapted cerrado")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
