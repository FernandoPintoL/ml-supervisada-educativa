"""
Módulo de autenticación para ML API

Proporciona validación de tokens Sanctum de Laravel
y control de acceso basado en roles.
"""

from .sanctum_auth import (
    SanctumAuthenticator,
    SanctumRoleValidator,
    TokenCache,
    init_sanctum_auth,
    get_authenticator,
    token_cache,
)

__all__ = [
    "SanctumAuthenticator",
    "SanctumRoleValidator",
    "TokenCache",
    "init_sanctum_auth",
    "get_authenticator",
    "token_cache",
]
