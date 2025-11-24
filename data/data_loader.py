"""
Data Loader para Plataforma Educativa
Plataforma Educativa ML

Carga datos desde PostgreSQL para entrenar modelos supervisados.
Conecta con tablas de:
- Users (estudiantes, docentes, padres)
- Calificaciones
- Asistencia
- Evaluaciones
- Cursos/Materias
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


class DataLoader:
    """
    Cargador de datos desde PostgreSQL.

    Proporciona métodos para:
    - Cargar datos de estudiantes
    - Cargar calificaciones
    - Cargar asistencia
    - Unir datos de múltiples tablas
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

        logger.info("✓ DataLoader inicializado")

    # ===========================================
    # MÉTODOS DE CARGA BÁSICA
    # ===========================================

    def load_students(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Cargar datos básicos de estudiantes.

        Intenta cargar estudiantes en este orden:
        1. Spatie Laravel Permissions (JOIN model_has_roles + roles)
        2. Columna 'role' directa en tabla users
        3. Fallback: cargar todos los usuarios

        Args:
            limit (int): Límite de registros (None = sin límite)

        Retorna:
            DataFrame: Estudiantes con columnas [id, name, email, role, created_at]
        """
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return pd.DataFrame()

            # Estrategia 1: Intentar con Spatie Laravel Permissions
            logger.info("Intentando carga con Spatie Laravel Permissions...")
            query_spatie = """
                SELECT DISTINCT
                    u.id,
                    u.name,
                    u.email,
                    'student' as role,
                    u.created_at
                FROM users u
                INNER JOIN model_has_roles mhr ON u.id = mhr.model_id
                INNER JOIN roles r ON mhr.role_id = r.id
                WHERE mhr.model_type = 'App\\\\Models\\\\User'
                  AND r.name = 'student'
            """

            if limit:
                query_spatie += f" LIMIT {limit}"

            try:
                df = pd.read_sql(query_spatie, self.db.bind)
                if not df.empty:
                    logger.info(f"✓ Cargados {len(df)} estudiantes (Spatie Permissions)")
                    return df
                else:
                    logger.info("No se encontraron estudiantes con Spatie Permissions")
            except Exception as e:
                logger.debug(f"Estrategia Spatie falló: {str(e)}")

            # Estrategia 2: Intentar con columna 'role' directa
            logger.info("Intentando carga con columna 'role' directa...")
            query_direct = """
                SELECT
                    id,
                    name,
                    email,
                    'student' as role,
                    created_at
                FROM users
                WHERE role = 'student'
            """

            if limit:
                query_direct += f" LIMIT {limit}"

            try:
                df = pd.read_sql(query_direct, self.db.bind)
                if not df.empty:
                    logger.info(f"✓ Cargados {len(df)} estudiantes (columna role directa)")
                    return df
                else:
                    logger.info("No se encontraron estudiantes con columna role")
            except Exception as e:
                logger.debug(f"Estrategia columna 'role' falló: {str(e)}")

            # Estrategia 3: Fallback - Cargar todos los usuarios
            logger.info("Usando fallback: cargando todos los usuarios...")
            query_fallback = """
                SELECT
                    id,
                    name,
                    email,
                    'unknown' as role,
                    created_at
                FROM users
                ORDER BY created_at DESC
            """

            if limit:
                query_fallback += f" LIMIT {limit}"

            df = pd.read_sql(query_fallback, self.db.bind)
            logger.warning(
                f"⚠ Cargados {len(df)} usuarios (fallback - estructura de BD no reconocida)"
            )
            return df

        except Exception as e:
            logger.error(f"✗ Error cargando estudiantes (todas las estrategias): {str(e)}")
            return pd.DataFrame()

    def load_grades(self, student_ids: Optional[List[int]] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """
        Cargar calificaciones de estudiantes desde tabla 'calificaciones'.

        Lee de la tabla calificaciones que agrupa notas por trabajo
        y las agrupa por estudiante para generar estadísticas.

        Args:
            student_ids (List[int]): IDs de estudiantes a filtrar (None = todos)
            limit (int): Límite de registros

        Retorna:
            DataFrame: Calificaciones [student_id, grade, fecha_calificacion]
        """
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return pd.DataFrame()

            # Consulta para la tabla 'calificaciones' con JOIN a 'trabajos'
            query = """
                SELECT
                    t.estudiante_id as student_id,
                    c.puntaje as grade,
                    c.fecha_calificacion
                FROM calificaciones c
                JOIN trabajos t ON c.trabajo_id = t.id
                WHERE t.estudiante_id IS NOT NULL
            """

            # Filtrar por estudiantes si se proporciona
            if student_ids:
                placeholders = ','.join(str(id) for id in student_ids)
                query += f" AND t.estudiante_id IN ({placeholders})"

            query += " ORDER BY c.fecha_calificacion DESC"

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql(query, self.db.bind)
            logger.info(f"✓ Cargadas {len(df)} calificaciones desde tabla 'calificaciones'")
            return df

        except Exception as e:
            logger.error(f"✗ Error cargando calificaciones: {str(e)}")
            return pd.DataFrame()

    def load_attendance(self, student_ids: Optional[List[int]] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """
        Cargar registro de asistencia desde tabla 'asistencias'.

        Lee la nueva tabla asistencias con estados: presente, ausente, tardanza, excused
        Convierte estados a valor binario (presente=1, otros=0)

        Args:
            student_ids (List[int]): IDs de estudiantes a filtrar
            limit (int): Límite de registros

        Retorna:
            DataFrame: Asistencia [student_id, present, fecha_asistencia]
        """
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return pd.DataFrame()

            # Consulta para la nueva tabla 'asistencias'
            query = """
                SELECT
                    estudiante_id as student_id,
                    CASE WHEN estado = 'presente' THEN 1 ELSE 0 END as present,
                    fecha_asistencia
                FROM asistencias
                WHERE estudiante_id IS NOT NULL
            """

            if student_ids:
                placeholders = ','.join(str(id) for id in student_ids)
                query += f" AND estudiante_id IN ({placeholders})"

            query += " ORDER BY fecha_asistencia DESC"

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql(query, self.db.bind)
            logger.info(f"✓ Cargados {len(df)} registros de asistencia desde tabla 'asistencias'")
            return df

        except Exception as e:
            logger.error(f"✗ Error cargando asistencia: {str(e)}")
            return pd.DataFrame()

    def load_activity(self, student_ids: Optional[List[int]] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """
        Cargar registro de actividad estudiantil desde tabla 'actividad_estudiante'.

        Lee tipos de actividad: login, leccion_vista, tarea_iniciada, tarea_completada,
        recurso_descargado, foro_participacion, prueba_realizada

        Args:
            student_ids (List[int]): IDs de estudiantes a filtrar
            limit (int): Límite de registros

        Retorna:
            DataFrame: Actividad [student_id, tipo_actividad, duracion_minutos, fecha_hora]
        """
        try:
            if self.db is None:
                logger.error("Sesión de BD no disponible")
                return pd.DataFrame()

            # Consulta para la nueva tabla 'actividad_estudiante'
            query = """
                SELECT
                    estudiante_id as student_id,
                    tipo_actividad,
                    duracion_minutos,
                    fecha_hora
                FROM actividad_estudiante
                WHERE estudiante_id IS NOT NULL
            """

            if student_ids:
                placeholders = ','.join(str(id) for id in student_ids)
                query += f" AND estudiante_id IN ({placeholders})"

            query += " ORDER BY fecha_hora DESC"

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql(query, self.db.bind)
            logger.info(f"✓ Cargados {len(df)} registros de actividad desde tabla 'actividad_estudiante'")
            return df

        except Exception as e:
            logger.error(f"✗ Error cargando actividad: {str(e)}")
            return pd.DataFrame()

    # ===========================================
    # MÉTODOS DE AGREGACIÓN Y FEATURE ENGINEERING
    # ===========================================

    def aggregate_student_stats(self, df_grades: pd.DataFrame,
                               df_attendance: pd.DataFrame,
                               df_activity: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Agregar estadísticas de estudiante desde calificaciones, asistencia y actividad.

        Crea features:
        - promedio_ultimas_notas (desde calificaciones)
        - varianza_notas (desde calificaciones)
        - asistencia_porcentaje (desde asistencias)
        - cantidad_asistencias, cantidad_inasistencias
        - horas_estudio_semanal (desde actividad_estudiante) ← NUEVO

        Args:
            df_grades (DataFrame): Calificaciones
            df_attendance (DataFrame): Asistencia
            df_activity (DataFrame): Actividad estudiantil (opcional)

        Retorna:
            DataFrame: Estadísticas agregadas por estudiante
        """
        try:
            if df_grades.empty and df_attendance.empty and (df_activity is None or df_activity.empty):
                logger.warning("DataFrames vacíos para agregación")
                return pd.DataFrame()

            # Agregar calificaciones
            if not df_grades.empty:
                grade_stats = df_grades.groupby('student_id')['grade'].agg([
                    ('promedio_ultimas_notas', 'mean'),
                    ('varianza_notas', 'var'),
                    ('max_nota', 'max'),
                    ('min_nota', 'min'),
                    ('cantidad_calificaciones', 'count')
                ]).reset_index()
            else:
                grade_stats = pd.DataFrame()

            # Agregar asistencia
            if not df_attendance.empty:
                attendance_stats = df_attendance.groupby('student_id')['present'].agg([
                    ('asistencia_porcentaje', 'mean'),
                    ('cantidad_asistencias', 'sum'),
                    ('cantidad_inasistencias', lambda x: (~x.astype(bool)).sum())
                ]).reset_index()
                # Convertir a porcentaje
                attendance_stats['asistencia_porcentaje'] = attendance_stats['asistencia_porcentaje'] * 100
            else:
                attendance_stats = pd.DataFrame()

            # Agregar actividad estudiantil (NUEVO)
            activity_stats = pd.DataFrame()
            if df_activity is not None and not df_activity.empty:
                # Calcular horas de estudio semanal
                # Considera solo actividades de aprendizaje (excluye login)
                learning_activities = df_activity[
                    df_activity['tipo_actividad'].isin([
                        'leccion_vista', 'tarea_completada',
                        'foro_participacion', 'prueba_realizada'
                    ])
                ].copy()

                if not learning_activities.empty:
                    # Sumar minutos por estudiante
                    total_minutos = learning_activities.groupby('student_id')['duracion_minutos'].sum()
                    # Convertir a horas
                    total_horas = total_minutos / 60
                    # Estimar horas semanales (dividir por número de semanas estimado: ~15 semanas en 3 meses)
                    horas_por_semana = total_horas / 15

                    activity_stats = pd.DataFrame({
                        'student_id': total_horas.index,
                        'horas_estudio_semanal': horas_por_semana.values
                    })

                    logger.info(f"✓ Horas de estudio calculadas para {len(activity_stats)} estudiantes")

            # Unir estadísticas paso a paso
            if not grade_stats.empty:
                result = grade_stats
                if not attendance_stats.empty:
                    result = result.merge(attendance_stats, on='student_id', how='outer')
            elif not attendance_stats.empty:
                result = attendance_stats
            else:
                result = pd.DataFrame()

            # Agregar actividad si existe
            if not activity_stats.empty and not result.empty:
                result = result.merge(activity_stats, on='student_id', how='outer')
            elif not activity_stats.empty:
                result = activity_stats

            # Llenar NaN con 0
            result = result.fillna(0)

            logger.info(f"✓ Estadísticas agregadas para {len(result)} estudiantes")
            return result

        except Exception as e:
            logger.error(f"✗ Error agregando estadísticas: {str(e)}")
            return pd.DataFrame()

    # ===========================================
    # MÉTODOS DE PREPARACIÓN COMPLETA
    # ===========================================

    def load_training_data(self, limit: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Cargar y preparar datos completos para entrenamiento.

        Proceso:
        1. Cargar estudiantes
        2. Cargar calificaciones, asistencia y actividad
        3. Agregar estadísticas
        4. Unir con datos de estudiantes

        Args:
            limit (int): Límite de estudiantes

        Retorna:
            Tuple[DataFrame, List[str]]:
            - DataFrame con datos listos para ML
            - Lista de nombres de features
        """
        try:
            logger.info("Iniciando carga de datos de entrenamiento...")

            # 1. Cargar datos base
            students = self.load_students(limit=limit)
            if len(students) < MIN_STUDENTS_REQUIRED:
                logger.warning(
                    f"Menos de {MIN_STUDENTS_REQUIRED} estudiantes disponibles. "
                    f"Actual: {len(students)}"
                )

            student_ids = students['id'].tolist()

            # 2. Cargar calificaciones, asistencia y actividad
            logger.info("Cargando calificaciones...")
            grades = self.load_grades(student_ids=student_ids)

            logger.info("Cargando asistencias...")
            attendance = self.load_attendance(student_ids=student_ids)

            logger.info("Cargando actividad estudiantil...")
            activity = self.load_activity(student_ids=student_ids)

            # 3. Agregar estadísticas (incluyendo datos de actividad)
            logger.info("Agregando estadísticas...")
            stats = self.aggregate_student_stats(grades, attendance, activity)

            # 4. Unir con estudiantes
            if not stats.empty:
                result = students.merge(stats, left_on='id', right_on='student_id', how='left')
                result = result.fillna(0)
                logger.info(f"✓ Datos de entrenamiento preparados: {result.shape}")
            else:
                result = students
                logger.warning("No se pudieron agregar estadísticas")

            # Features disponibles
            features = [col for col in result.columns if col not in ['id', 'name', 'email', 'role', 'created_at', 'student_id']]

            logger.info(f"✓ Features disponibles: {features}")
            return result, features

        except Exception as e:
            logger.error(f"✗ Error cargando datos de entrenamiento: {str(e)}")
            return pd.DataFrame(), []

    # ===========================================
    # UTILIDADES
    # ===========================================

    def test_connection(self) -> bool:
        """
        Probar conexión a base de datos.

        Retorna:
            bool: True si la conexión es exitosa
        """
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
        """Cerrar sesión de base de datos si fue creada internamente."""
        if self.own_session and self.db:
            self.db.close()
            logger.info("DataLoader cerrado")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
