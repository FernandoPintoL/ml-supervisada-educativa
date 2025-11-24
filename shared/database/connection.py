"""
Módulo de Conexión a Base de Datos PostgreSQL
Plataforma Educativa ML

Proporciona:
- SQLAlchemy Engine y Session Factory
- Connection pooling
- Manejo de conexiones con try/except
"""

import os
from typing import Generator, Optional
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import logging

# Importar configuración
from shared.config import (
    DATABASE_URL,
    DATABASE_HOST,
    DATABASE_PORT,
    DATABASE_NAME,
    DATABASE_USER,
    DATABASE_PASSWORD,
    DEBUG
)

# Configurar logger
logger = logging.getLogger(__name__)

# ===========================================
# CREAR ENGINE DE SQLALCHEMY
# ===========================================

def create_db_engine():
    """
    Crea el engine de SQLAlchemy con pool de conexiones.

    Retorna:
        Engine: SQLAlchemy Engine configurado
    """
    try:
        # Construir URL si las variables individuales están disponibles
        if all([DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD]):
            db_url = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
        else:
            db_url = DATABASE_URL

        if DEBUG:
            logger.info(f"Conectando a base de datos: {db_url}")

        # Crear engine con pool de conexiones
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=10,  # Número de conexiones en el pool
            max_overflow=20,  # Conexiones adicionales permitidas
            pool_recycle=3600,  # Reciclar conexiones cada hora
            echo=DEBUG,  # Log SQL queries en debug
            connect_args={
                'connect_timeout': 10,
                'application_name': 'plataforma_educativa_ml'
            }
        )

        # Event listener para logging
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            if DEBUG:
                logger.debug("Nueva conexión a base de datos establecida")

        @event.listens_for(engine, "close")
        def receive_close(dbapi_conn, connection_record):
            if DEBUG:
                logger.debug("Conexión a base de datos cerrada")

        logger.info("✓ Engine de base de datos creado exitosamente")
        return engine

    except Exception as e:
        logger.error(f"✗ Error creando engine de base de datos: {str(e)}")
        raise


# Crear engine global
try:
    engine = create_db_engine()
except Exception as e:
    logger.error(f"Error fatal al conectar a la base de datos: {str(e)}")
    engine = None


# ===========================================
# SESSION FACTORY
# ===========================================

# Crear SessionLocal factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency para FastAPI que proporciona una sesión de BD.

    Uso en FastAPI:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...

    Yields:
        Session: Sesión de SQLAlchemy
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Error en sesión de BD: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


# ===========================================
# FUNCIONES DE UTILIDAD
# ===========================================

def test_connection() -> bool:
    """
    Prueba la conexión a la base de datos.

    Retorna:
        bool: True si la conexión es exitosa, False otherwise
    """
    if engine is None:
        logger.error("Engine no está inicializado")
        return False

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✓ Conexión a base de datos verificada")
            return True
    except Exception as e:
        logger.error(f"✗ Error probando conexión: {str(e)}")
        return False


def get_db_session() -> Optional[Session]:
    """
    Obtiene una sesión de base de datos (sin usar FastAPI Depends).

    Uso para scripts:
        db = get_db_session()
        if db:
            # hacer cosas
            db.close()

    Retorna:
        Session o None si hay error
    """
    try:
        return SessionLocal()
    except Exception as e:
        logger.error(f"Error creando sesión: {str(e)}")
        return None


def get_db_connection():
    """
    Alias para get_db_session() - Obtiene una conexión/sesión de base de datos.

    Uso:
        conn = get_db_connection()
        if conn:
            # hacer consultas
            conn.close()

    Retorna:
        Session o None si hay error
    """
    return get_db_session()


def close_db() -> None:
    """
    Cierra todas las conexiones del pool.

    Uso:
        close_db()  # Llamar al finalizar la aplicación
    """
    if engine:
        engine.dispose()
        logger.info("Pool de conexiones cerrado")


# ===========================================
# CONTEXTO MANAGER PARA SCRIPTS
# ===========================================

class DBSession:
    """
    Context manager para manejar sesiones en scripts.

    Uso:
        with DBSession() as db:
            # hacer cosas
    """

    def __init__(self):
        self.db = None

    def __enter__(self) -> Session:
        self.db = SessionLocal()
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            if exc_type:
                self.db.rollback()
                logger.error(f"Rollback ejecutado: {exc_val}")
            else:
                self.db.commit()
            self.db.close()
