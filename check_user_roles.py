#!/usr/bin/env python
"""
Verificación de diversidad de roles en tabla users
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env
env_path = Path(__file__).parent.parent / 'supervisado' / '.env'
load_dotenv(env_path)

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_DATABASE', 'educativa')
DB_USER = os.getenv('DB_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '1234')

try:
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    print("=" * 70)
    print("VALIDACION DE ROLES EN BD - TABLA USERS")
    print("=" * 70)

    # 1. Contar usuarios por rol
    query_roles = """
    SELECT tipo_usuario, COUNT(*) as cantidad
    FROM users
    GROUP BY tipo_usuario
    ORDER BY cantidad DESC
    """
    cursor.execute(query_roles)
    resultados = cursor.fetchall()

    print("\n[OK] DISTRIBUCION DE ROLES:")
    print("-" * 70)
    total_usuarios = sum([r['cantidad'] for r in resultados])
    for row in resultados:
        role = row['tipo_usuario']
        cantidad = row['cantidad']
        porcentaje = (cantidad / total_usuarios) * 100
        print(f"  {role:15} | {cantidad:4} usuarios | {porcentaje:5.1f}%")

    print("-" * 70)
    print(f"  TOTAL            | {total_usuarios:4} usuarios | 100.0%")

    # 2. Ejemplos de cada rol
    print("\n[OK] EJEMPLOS POR CADA ROL:")
    print("-" * 70)

    for row in resultados:
        role = row['tipo_usuario']
        query_examples = """
        SELECT id, name, apellido, email, tipo_usuario
        FROM users
        WHERE tipo_usuario = %s
        LIMIT 2
        """
        cursor.execute(query_examples, (role,))
        examples = cursor.fetchall()

        print(f"\n  {role.upper()}:")
        for idx, ex in enumerate(examples, 1):
            nombre_completo = f"{ex['name']} {ex['apellido']}" if ex['apellido'] else ex['name']
            print(f"    {idx}. [{ex['id']}] {nombre_completo} ({ex['email']})")

    # 3. Estadísticas de completitud
    print("\n[OK] VALIDACION DE DATOS:")
    print("-" * 70)

    query_stats = """
    SELECT
        COUNT(*) as total,
        COUNT(CASE WHEN name IS NOT NULL THEN 1 END) as con_nombre,
        COUNT(CASE WHEN email IS NOT NULL THEN 1 END) as con_email,
        COUNT(CASE WHEN tipo_usuario IS NOT NULL THEN 1 END) as con_rol
    FROM users
    """
    cursor.execute(query_stats)
    stats = cursor.fetchone()

    print(f"  Total usuarios:           {stats['total']}")
    print(f"  Con nombre:               {stats['con_nombre']:4} ({(stats['con_nombre']/stats['total'])*100:.1f}%)")
    print(f"  Con email:                {stats['con_email']:4} ({(stats['con_email']/stats['total'])*100:.1f}%)")
    print(f"  Con rol asignado:         {stats['con_rol']:4} ({(stats['con_rol']/stats['total'])*100:.1f}%)")

    # 4. Validación de estudiantes específicamente
    print("\n[OK] VALIDACION DE ESTUDIANTES:")
    print("-" * 70)
    query_students = """
    SELECT
        COUNT(*) as total_estudiantes,
        COUNT(CASE WHEN desempeño_promedio IS NOT NULL THEN 1 END) as con_desempeño,
        COUNT(CASE WHEN asistencia_porcentaje IS NOT NULL THEN 1 END) as con_asistencia,
        COUNT(CASE WHEN participacion_porcentaje IS NOT NULL THEN 1 END) as con_participacion,
        COUNT(CASE WHEN tareas_completadas IS NOT NULL THEN 1 END) as con_tareas
    FROM users
    WHERE tipo_usuario = 'estudiante'
    """
    cursor.execute(query_students)
    student_stats = cursor.fetchone()

    print(f"  Total estudiantes:        {student_stats['total_estudiantes']}")
    print(f"  Con desempeño:            {student_stats['con_desempeño']:4} ({(student_stats['con_desempeño']/max(1,student_stats['total_estudiantes']))*100:.1f}%)")
    print(f"  Con asistencia:           {student_stats['con_asistencia']:4} ({(student_stats['con_asistencia']/max(1,student_stats['total_estudiantes']))*100:.1f}%)")
    print(f"  Con participación:        {student_stats['con_participacion']:4} ({(student_stats['con_participacion']/max(1,student_stats['total_estudiantes']))*100:.1f}%)")
    print(f"  Con tareas:               {student_stats['con_tareas']:4} ({(student_stats['con_tareas']/max(1,student_stats['total_estudiantes']))*100:.1f}%)")

    print("\n" + "=" * 70)
    print("[DONE] VALIDACION COMPLETADA")
    print("=" * 70)

    cursor.close()
    conn.close()

except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
