# 🚀 Guía: Entrenar Modelos ML desde supervisado/

Este directorio es **completamente independiente**. Todo lo necesario para entrenar los modelos está aquí.

---

## 📋 Pre-requisitos

1. **PostgreSQL en ejecución** (con la BD de Laravel)
2. **Python 3.8+** instalado
3. **Virtual Environment activado**
4. **Archivo `.env` configurado**

---

## ⚙️ Verificar Configuración

### 1. Verificar archivo `.env`

```powershell
cat .env
```

Debe contener:
```env
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=educativa
DB_USERNAME=postgres
DB_PASSWORD=1234
```

### 2. Verificar conexión a BD (Opcional)

```powershell
python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        database='educativa',
        user='postgres',
        password='1234'
    )
    print('[OK] Conexion a BD exitosa')
    conn.close()
except Exception as e:
    print(f'[ERROR] {e}')
"
```

---

## 🎯 Entrenar Modelos

### Opción 1: Entrenar TODOS los modelos (RECOMENDADO)

```powershell
# Desde supervisado/
cd "D:\PLATAFORMA EDUCATIVA\supervisado"

# Activar virtual environment
venv\Scripts\Activate

# Ejecutar entrenamiento
python train_models_simple.py
```

**Salida esperada:**
```
======================================================================
INICIANDO ENTRENAMIENTO DE MODELOS ML
======================================================================
[*] Cargando datos de la base de datos...
[OK] Datos cargados: 100 estudiantes
[MODEL] Entrenando Performance Predictor...
[OK] Modelo entrenado:
  RMSE: 18.5432
  R2: 0.7234
  Modelo guardado: trained_models\PerformancePredictor_model.pkl
[MODEL] Entrenando Career Recommender...
[OK] Modelo entrenado:
  Accuracy: 0.8500
  Modelo guardado: trained_models\CareerRecommender_model.pkl
[MODEL] Entrenando Trend Predictor...
[OK] Modelo entrenado:
  Accuracy: 0.7800
  Modelo guardado: trained_models\TrendPredictor_model.pkl
[MODEL] Entrenando Progress Analyzer...
[OK] Modelo entrenado:
  R2: 0.6950
  Modelo guardado: trained_models\ProgressAnalyzer_model.pkl

======================================================================
RESUMEN DE ENTRENAMIENTOS
======================================================================
[OK] - Performance Predictor
[OK] - Career Recommender
[OK] - Trend Predictor
[OK] - Progress Analyzer
======================================================================

[SUCCESS] Todos los modelos entrenados exitosamente!
Modelos guardados en: D:\PLATAFORMA EDUCATIVA\supervisado\trained_models

Archivos generados:
  - PerformancePredictor_model.pkl
  - CareerRecommender_model.pkl
  - TrendPredictor_model.pkl
  - ProgressAnalyzer_model.pkl
```

### Opción 2: Entrenar modelos individuales (en desarrollo)

```powershell
# Próximamente disponible
python training/train_performance_adapted.py
python training/train_career_recommender.py
python training/train_trend_predictor.py
python training/train_progress_analyzer.py
```

---

## ✅ Verificar Modelos Entrenados

### Comprobar que los archivos existen

```powershell
# Desde supervisado/
ls trained_models/

# Deberías ver:
# - PerformancePredictor_model.pkl
# - CareerRecommender_model.pkl
# - TrendPredictor_model.pkl
# - ProgressAnalyzer_model.pkl
# - training_log.json
```

### Ver detalles del entrenamiento

```powershell
# Ver log de entrenamientos
cat trained_models/training_log.json
```

---

## 🚀 Usar los Modelos en Predicciones

Una vez entrenados los modelos, puedes:

1. **Iniciar servidor API** (en terminal separada)
   ```powershell
   python -m uvicorn api_server:app --port 8001
   ```

2. **Hacer predicciones** (desde otra terminal)
   ```powershell
   # Test de predicción de riesgo
   curl -X POST http://localhost:8001/predict/risk `
     -H "Content-Type: application/json" `
     -d '{"student_id": 1}'
   ```

---

## 📁 Estructura de Directorios (Independiente)

```
supervisado/
├── train_models_simple.py          # [NUEVO] Script simple de entrenamiento
├── setup_env.py                    # [NUEVO] Script de configuración
├── EJECUTAR_ENTRENAMIENTO.md       # [NUEVO] Esta guía
├── shared/                         # [COPIADO de ml_educativas]
│   ├── database/
│   │   └── connection.py
│   ├── config.py
│   └── ...
├── trained_models/                 # [OUTPUT] Modelos entrenados
│   ├── PerformancePredictor_model.pkl
│   ├── CareerRecommender_model.pkl
│   ├── TrendPredictor_model.pkl
│   ├── ProgressAnalyzer_model.pkl
│   └── training_log.json
├── training/                       # Scripts de entrenamiento
├── api_server.py                  # Servidor API
├── api/                           # Endpoints
├── venv/                          # Virtual environment
└── ...
```

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'shared'"
**Solución:** Ejecutar `python setup_env.py` en supervisado/

### Error: "Connection refused" (BD)
**Solución:** Verificar que PostgreSQL está corriendo:
```powershell
# Windows
pg_ctl -D "C:\Program Files\PostgreSQL\data" status

# Si no está corriendo:
pg_ctl -D "C:\Program Files\PostgreSQL\data" start
```

### Error: "No hay datos disponibles"
**Solución:** Verificar que ejecutaste los seeders en Laravel:
```powershell
cd D:\PLATAFORMA EDUCATIVA\plataforma-educativa
php artisan migrate:fresh --seed
```

### Modelo toma mucho tiempo
**Normal:** Entrenar puede tomar 30 segundos a 2 minutos
**Si toma >5min:** Revisa uso de CPU/memoria o reduce LIMIT en query

---

## 📊 Próximos Pasos

1. ✅ Entrenar modelos
2. ✅ Verificar archivos en trained_models/
3. ⬜ Iniciar servidor API
4. ⬜ Integrar predicciones en Laravel
5. ⬜ Crear dashboard de monitoreo

---

## 📝 Notas

- **Todos los modelos se entrenan con datos de la BD de Laravel**
- **Los modelos se guardan en `trained_models/`**
- **No hay dependencia de `ml_educativas/` o `no_supervisado/`**
- **Todo es independiente en `supervisado/`**

---

**Última actualización:** 25 de Noviembre 2025
**Estado:** ✅ Listo para usar
