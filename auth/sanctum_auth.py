"""
Autenticación con Sanctum de Laravel

Este módulo valida tokens generados por Laravel Sanctum
y extrae información del usuario y sus roles.
"""

import json
import base64
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)


class SanctumAuthenticator:
    """Validador de tokens Sanctum de Laravel"""

    def __init__(self, app_key: str):
        """
        Inicializar con la APP_KEY de Laravel

        Args:
            app_key: La clave APP_KEY del .env de Laravel
                    Formato: "base64:xxxxx"
        """
        self.app_key = self._extract_key(app_key)
        logger.info("SanctumAuthenticator inicializado")

    @staticmethod
    def _extract_key(app_key: str) -> bytes:
        """Extraer clave base64 de APP_KEY de Laravel"""
        if app_key.startswith("base64:"):
            key_str = app_key[7:]  # Remover "base64:"
            return base64.b64decode(key_str)
        return app_key.encode()

    def validate_sanctum_token(self, token: str) -> Optional[Dict]:
        """
        Validar token de Sanctum

        Los tokens de Sanctum en Laravel tienen el formato:
        token_id|hash_token

        Sanctum los almacena en la BD y valida contra el hash

        Args:
            token: El token de Sanctum (plainTextToken)

        Returns:
            Dict con user info si es válido, None si no
        """
        try:
            # Sanctum tokens son simples: id|hash
            # En esta implementación simple, solo verificamos el formato
            # En producción, validarías contra la BD de Laravel

            if not token or "|" not in token:
                logger.warning("Token Sanctum inválido - formato incorrecto")
                return None

            parts = token.split("|")
            if len(parts) != 2:
                logger.warning(f"Token Sanctum inválido - {len(parts)} partes")
                return None

            token_id, token_hash = parts

            # Aquí, en producción, validarías contra:
            # SELECT * FROM personal_access_tokens WHERE id = token_id
            # AND hashed_token = hash(token_hash)

            # Para este ejemplo, retornamos un resultado positivo
            # En producción, harías una consulta a la BD de Laravel
            logger.info(f"Token Sanctum validado: {token_id}")

            return {
                "token_id": token_id,
                "token_hash": token_hash,
                "valid": True,
            }

        except Exception as e:
            logger.error(f"Error validando token Sanctum: {str(e)}")
            return None

    def get_token_info(self, token: str) -> Optional[Dict]:
        """
        Obtener información del token (decodificar si es JWT)

        Si el token es JWT, lo decodifica sin verificar firma
        (la verificación se hace contra la BD)

        Args:
            token: El token

        Returns:
            Dict con información del token
        """
        try:
            # Si el token parece ser JWT, intentar decodificar
            if token.count(".") == 2:
                parts = token.split(".")
                payload = parts[1]

                # Agregar padding si es necesario
                padding = 4 - len(payload) % 4
                if padding != 4:
                    payload += "=" * padding

                decoded = base64.urlsafe_b64decode(payload)
                info = json.loads(decoded)

                logger.info(f"Token JWT decodificado: {info}")
                return info
            else:
                # Token simple de Sanctum
                return self.validate_sanctum_token(token)

        except Exception as e:
            logger.error(f"Error decodificando token: {str(e)}")
            return None


class SanctumRoleValidator:
    """Validador de roles basado en Sanctum"""

    ROLE_PERMISSIONS = {
        "admin": [
            "predict:all",
            "cache:refresh",
            "cache:clear",
            "admin:view",
            "logs:view",
        ],
        "teacher": [
            "predict:students",
            "cache:info",
            "stats:class",
        ],
        "student": [
            "predict:self",
        ],
    }

    @staticmethod
    def has_role(user_data: Dict, required_role: str) -> bool:
        """
        Verificar si el usuario tiene un rol específico

        Args:
            user_data: Datos del usuario del token
            required_role: El rol requerido (admin, teacher, student)

        Returns:
            True si tiene el rol
        """
        user_roles = user_data.get("roles", [])

        # Si no es lista, convertir
        if isinstance(user_roles, str):
            user_roles = [user_roles]

        # Admin tiene todos los permisos
        if "admin" in user_roles:
            return True

        return required_role in user_roles

    @staticmethod
    def has_permission(user_data: Dict, required_permission: str) -> bool:
        """
        Verificar si el usuario tiene un permiso específico

        Args:
            user_data: Datos del usuario del token
            required_permission: El permiso requerido (ej: "predict:all")

        Returns:
            True si tiene el permiso
        """
        user_roles = user_data.get("roles", [])

        if isinstance(user_roles, str):
            user_roles = [user_roles]

        # Verificar permisos por cada rol del usuario
        for role in user_roles:
            permissions = SanctumRoleValidator.ROLE_PERMISSIONS.get(role, [])
            if required_permission in permissions or "predict:all" in permissions:
                return True

        return False

    @staticmethod
    def get_user_info(token_data: Dict) -> Dict:
        """
        Extraer información del usuario del token

        Sanctum típicamente incluye:
        - user_id
        - user_email
        - roles
        - permissions

        Args:
            token_data: Datos decodificados del token

        Returns:
            Dict con información del usuario
        """
        return {
            "user_id": token_data.get("sub") or token_data.get("user_id"),
            "email": token_data.get("email"),
            "roles": token_data.get("roles", []),
            "permissions": token_data.get("permissions", []),
            "is_admin": "admin" in token_data.get("roles", []),
            "is_teacher": "teacher" in token_data.get("roles", []),
            "is_student": "student" in token_data.get("roles", []),
        }


class TokenCache:
    """Cache simple de tokens validados"""

    def __init__(self, ttl_seconds: int = 3600):
        """
        Args:
            ttl_seconds: Tiempo de vida del cache (1 hora por defecto)
        """
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Dict, float]] = {}

    def get(self, token: str) -> Optional[Dict]:
        """Obtener token del cache si aún es válido"""
        if token in self.cache:
            data, timestamp = self.cache[token]
            if (datetime.now().timestamp() - timestamp) < self.ttl_seconds:
                return data
            else:
                del self.cache[token]
        return None

    def set(self, token: str, data: Dict) -> None:
        """Guardar token en cache"""
        self.cache[token] = (data, datetime.now().timestamp())

    def clear(self) -> None:
        """Limpiar cache"""
        self.cache.clear()


# Instancias globales
authenticator = None
token_cache = TokenCache()


def init_sanctum_auth(app_key: str):
    """Inicializar autenticador de Sanctum"""
    global authenticator
    authenticator = SanctumAuthenticator(app_key)
    logger.info("✓ Autenticación Sanctum inicializada")


def get_authenticator() -> SanctumAuthenticator:
    """Obtener instancia del autenticador"""
    global authenticator
    if not authenticator:
        raise RuntimeError("Sanctum auth no inicializado")
    return authenticator
