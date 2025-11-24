"""
Generador de Datos de Prueba para ML
Plataforma Educativa ML

Genera datos ficticios para pruebas de modelos de ML.
Los datos se insertan directamente en la BD PostgreSQL.

Uso (desde ml_educativas/):
    python -m supervisado.data.seed_test_data
    python -m supervisado.data.seed_test_data --students 100 --grades-per-student 15
    python -m supervisado.data.seed_test_data --clean  # Limpiar datos previos

Uso (desde cualquier lado):
    python ml_educativas/supervisado/data/seed_test_data.py
"""

import sys
import os
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

# Agregar ml_educativas al path
current_file = os.path.abspath(__file__)
data_dir = os.path.dirname(current_file)
supervisado_dir = os.path.dirname(os.path.dirname(current_file))
ml_educativas_dir = os.path.dirname(supervisado_dir)

if ml_educativas_dir not in sys.path:
    sys.path.insert(0, ml_educativas_dir)

from shared.database.connection import get_db_session, DBSession
from shared.config import DEBUG, LOG_LEVEL

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestDataSeeder:
    """Generador de datos de prueba para ML."""

    def __init__(self):
        """Inicializar seeder."""
        self.db = None
        logger.info("✓ TestDataSeeder inicializado")

    def connect(self) -> bool:
        """Conectar a la base de datos."""
        try:
            self.db = get_db_session()
            if self.db is None:
                logger.error("No se pudo obtener sesión de BD")
                return False

            # Test connection
            self.db.execute(text("SELECT 1"))
            logger.info("✓ Conectado a BD")
            return True
        except Exception as e:
            logger.error(f"✗ Error conectando: {str(e)}")
            return False

    def _get_students_from_db(self) -> List[int]:
        """Obtener IDs de estudiantes existentes desde BD."""
        try:
            # Estrategia 1: Spatie Permissions
            logger.info("Intentando obtener estudiantes con Spatie...")
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
                    logger.info(f"✓ Encontrados {len(student_ids)} estudiantes con Spatie")
                    return student_ids
            except Exception as e:
                logger.debug(f"Estrategia Spatie falló: {str(e)}")
                self.db.rollback()

            # Estrategia 2: Columna role directa
            logger.info("Intentando obtener estudiantes con columna role...")
            query = "SELECT id FROM users WHERE role = 'student'"
            try:
                result = self.db.execute(text(query))
                self.db.commit()
                student_ids = [row[0] for row in result.fetchall()]
                if student_ids:
                    logger.info(f"✓ Encontrados {len(student_ids)} estudiantes (columna role)")
                    return student_ids
            except Exception as e:
                logger.debug(f"Estrategia columna 'role' falló: {str(e)}")
                self.db.rollback()

            # Fallback: retornar primeros 20 usuarios
            logger.warning("⚠ No se encontraron estudiantes, usando primeros 20 usuarios")
            query = "SELECT id FROM users LIMIT 20"
            result = self.db.execute(text(query))
            self.db.commit()
            student_ids = [row[0] for row in result.fetchall()]

            if student_ids:
                return student_ids
            else:
                logger.error("No hay usuarios en la base de datos")
                return []

        except Exception as e:
            logger.error(f"✗ Error obteniendo estudiantes: {str(e)}", exc_info=True)
            try:
                self.db.rollback()
            except:
                pass
            return []

    def seed_grades(self, num_grades: int = 1000,
                    grades_per_student: int = 15,
                    clean_first: bool = False) -> bool:
        """
        Generar calificaciones ficticias.

        Args:
            num_grades (int): Número total de calificaciones a generar
            grades_per_student (int): Aproximado de calificaciones por estudiante
            clean_first (bool): Limpiar datos previos primero

        Retorna:
            bool: True si exitoso
        """
        try:
            logger.info("\n[1/4] Generando calificaciones...")

            # Obtener IDs de estudiantes
            student_ids = self._get_students_from_db()
            if not student_ids:
                logger.error("No hay estudiantes disponibles")
                return False

            # Limpiar si se pide
            if clean_first:
                logger.info("Limpiando calificaciones previas...")
                self.db.execute(text("DELETE FROM grades"))
                self.db.commit()

            # Temas ejemplo
            subjects = ['Matemática', 'Español', 'Ciencias', 'Historia', 'Inglés',
                       'Física', 'Química', 'Biología', 'Educación Física', 'Arte']

            # Generar calificaciones
            grades_to_insert = []
            now = datetime.now()

            for i in range(num_grades):
                student_id = np.random.choice(student_ids)
                grade = np.random.normal(loc=7.0, scale=1.5)  # Media 7, desviación 1.5
                grade = np.clip(grade, 1.0, 10.0)  # Limitar entre 1 y 10

                subject = np.random.choice(subjects)
                evaluated_at = now - timedelta(days=np.random.randint(0, 180))

                grades_to_insert.append({
                    'student_id': int(student_id),
                    'subject': subject,
                    'grade': float(round(grade, 2)),
                    'evaluated_at': evaluated_at
                })

            # Insertar en lotes
            batch_size = 100
            for i in range(0, len(grades_to_insert), batch_size):
                batch = grades_to_insert[i:i+batch_size]

                for record in batch:
                    insert_query = f"""
                        INSERT INTO grades (student_id, subject, grade, evaluated_at, created_at, updated_at)
                        VALUES ({record['student_id']}, '{record['subject']}', {record['grade']},
                                '{record['evaluated_at'].isoformat()}', NOW(), NOW())
                    """
                    try:
                        self.db.execute(text(insert_query))
                    except Exception as e:
                        logger.debug(f"Saltar registro duplicado: {str(e)}")

            self.db.commit()
            logger.info(f"✓ Insertadas {len(grades_to_insert)} calificaciones")
            return True

        except Exception as e:
            logger.error(f"✗ Error en seed_grades: {str(e)}", exc_info=True)
            if self.db:
                self.db.rollback()
            return False

    def seed_attendance(self, num_records: int = 1500,
                       clean_first: bool = False) -> bool:
        """
        Generar registro de asistencia ficticio.

        Args:
            num_records (int): Número de registros de asistencia
            clean_first (bool): Limpiar datos previos primero

        Retorna:
            bool: True si exitoso
        """
        try:
            logger.info("\n[2/4] Generando asistencia...")

            # Obtener IDs de estudiantes
            student_ids = self._get_students_from_db()
            if not student_ids:
                logger.error("No hay estudiantes disponibles")
                return False

            # Limpiar si se pide
            if clean_first:
                logger.info("Limpiando asistencia previa...")
                self.db.execute(text("DELETE FROM attendance"))
                self.db.commit()

            # Generar registros
            attendance_to_insert = []
            now = datetime.now()

            for i in range(num_records):
                student_id = np.random.choice(student_ids)
                # 85% de probabilidad de asistencia
                present = np.random.random() < 0.85
                attended_at = now - timedelta(days=np.random.randint(0, 180))

                attendance_to_insert.append({
                    'student_id': int(student_id),
                    'present': present,
                    'attended_at': attended_at
                })

            # Insertar en lotes
            batch_size = 100
            for i in range(0, len(attendance_to_insert), batch_size):
                batch = attendance_to_insert[i:i+batch_size]

                for record in batch:
                    present_val = 'true' if record['present'] else 'false'
                    insert_query = f"""
                        INSERT INTO attendance (student_id, present, attended_at, created_at, updated_at)
                        VALUES ({record['student_id']}, {present_val},
                                '{record['attended_at'].isoformat()}', NOW(), NOW())
                    """
                    try:
                        self.db.execute(text(insert_query))
                    except Exception as e:
                        logger.debug(f"Saltar registro duplicado: {str(e)}")

            self.db.commit()
            logger.info(f"✓ Insertados {len(attendance_to_insert)} registros de asistencia")
            return True

        except Exception as e:
            logger.error(f"✗ Error en seed_attendance: {str(e)}", exc_info=True)
            if self.db:
                self.db.rollback()
            return False

    def verify_data(self) -> bool:
        """Verificar que hay suficientes datos."""
        try:
            logger.info("\n[3/4] Verificando datos...")

            # Contar registros
            student_count = self.db.execute(text("SELECT COUNT(*) FROM users")).scalar()
            self.db.commit()

            # Intentar contar calificaciones desde tabla correcta
            try:
                calificaciones_count = self.db.execute(text("SELECT COUNT(*) FROM calificaciones")).scalar()
                self.db.commit()
            except:
                self.db.rollback()
                calificaciones_count = 0

            # Intentar contar asistencia
            try:
                attendance_count = self.db.execute(text("SELECT COUNT(*) FROM attendance")).scalar()
                self.db.commit()
            except:
                self.db.rollback()
                attendance_count = 0

            logger.info(f"✓ Usuarios: {student_count}")
            logger.info(f"✓ Calificaciones (calificaciones table): {calificaciones_count}")
            logger.info(f"✓ Registros de asistencia: {attendance_count}")

            if attendance_count < 100:
                logger.warning("⚠ Menos de 100 registros de asistencia. Se recomienda más datos.")
                return False

            return True

        except Exception as e:
            logger.error(f"✗ Error verificando datos: {str(e)}")
            try:
                self.db.rollback()
            except:
                pass
            return False

    def run(self, num_grades: int = 1000,
            grades_per_student: int = 15,
            num_attendance: int = 1500,
            clean_first: bool = False) -> bool:
        """
        Ejecutar seeding completo.

        Args:
            num_grades (int): Número de calificaciones a generar
            grades_per_student (int): Aproximado por estudiante
            num_attendance (int): Número de registros de asistencia
            clean_first (bool): Limpiar datos previos

        Retorna:
            bool: True si exitoso
        """
        try:
            logger.info("="*60)
            logger.info("TEST DATA SEEDER - PLATAFORMA EDUCATIVA ML")
            logger.info("="*60)

            # Conectar
            if not self.connect():
                return False

            # Sembrar datos
            if not self.seed_grades(num_grades, grades_per_student, clean_first):
                return False

            if not self.seed_attendance(num_attendance, clean_first):
                return False

            # Verificar
            if not self.verify_data():
                logger.warning("⚠ Verificación incompleta, pero continuando")

            logger.info("\n[4/4] Seeding completado")
            logger.info("="*60)
            logger.info("✓ DATOS DE PRUEBA GENERADOS EXITOSAMENTE")
            logger.info("="*60)

            return True

        except Exception as e:
            logger.error(f"✗ Error en seeding: {str(e)}", exc_info=True)
            return False

        finally:
            if self.db:
                self.db.close()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Generar datos de prueba para ML'
    )
    parser.add_argument(
        '--students',
        type=int,
        default=None,
        help='Número aproximado de estudiantes'
    )
    parser.add_argument(
        '--grades',
        type=int,
        default=1000,
        help='Número de calificaciones a generar (default: 1000)'
    )
    parser.add_argument(
        '--grades-per-student',
        type=int,
        default=15,
        help='Aproximado de calificaciones por estudiante'
    )
    parser.add_argument(
        '--attendance',
        type=int,
        default=1500,
        help='Número de registros de asistencia (default: 1500)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Limpiar datos previos antes de generar'
    )

    args = parser.parse_args()

    # Crear y ejecutar seeder
    seeder = TestDataSeeder()
    success = seeder.run(
        num_grades=args.grades,
        grades_per_student=args.grades_per_student,
        num_attendance=args.attendance,
        clean_first=args.clean
    )

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
