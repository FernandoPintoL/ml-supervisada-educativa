#!/usr/bin/env python
"""
Script para configurar el entorno de supervisado de forma independiente
Copia los módulos compartidos y verifica la estructura
"""

import shutil
import os
from pathlib import Path

def main():
    # Directorios
    supervisado_dir = Path(__file__).parent
    ml_educativas_dir = supervisado_dir.parent / 'ml_educativas'
    shared_src = ml_educativas_dir / 'shared'
    shared_dst = supervisado_dir / 'shared'

    print("=" * 60)
    print("CONFIGURANDO ENTORNO DE SUPERVISADO")
    print("=" * 60)

    # 1. Copiar shared si no existe
    if not shared_dst.exists():
        if shared_src.exists():
            print("[*] Copiando shared de " + str(shared_src))
            shutil.copytree(shared_src, shared_dst)
            print("[OK] Shared copiado exitosamente")
        else:
            print("[ERROR] No se encontró shared en ml_educativas")
            return False
    else:
        print("[OK] Shared ya existe")

    # 2. Crear directorios necesarios
    dirs_to_create = [
        supervisado_dir / 'trained_models',
        supervisado_dir / 'logs',
        supervisado_dir / 'data',
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(exist_ok=True)

    print("[OK] Directorios verificados/creados")

    # 3. Verificar estructura
    print("\nEstructura de supervisado:")
    for item in supervisado_dir.iterdir():
        if item.is_dir() and not item.name.startswith(('.', '__')):
            print("  [DIR] " + item.name)
        elif item.is_file() and (item.suffix in ['.py', '.md', '.txt', '.env']):
            print("  [FILE] " + item.name)

    print("\n[SUCCESS] Entorno configurado correctamente")
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
