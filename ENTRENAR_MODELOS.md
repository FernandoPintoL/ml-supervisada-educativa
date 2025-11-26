# 🎓 Guía de Entrenamientos - Modelos ML Supervisados

Pasos para entrenar cada modelo de Machine Learning.

---

## 📋 Modelos a Entrenar

| Modelo | Script | Puerto | Archivo Salida |
|--------|--------|--------|-----------------|
| 1️⃣ Performance Predictor | `train_performance_adapted.py` | 8001 | `PerformancePredictor_model.pkl` |
| 2️⃣ Career Recommender | `train_career_recommender.py` | 8001 | `CareerRecommender_model.pkl` |
| 3️⃣ Trend Predictor | `train_trend_predictor.py` | 8001 | `TrendPredictor_model.pkl` |
| 4️⃣ Progress Analyzer | `train_progress_analyzer.py` | 8001 | `ProgressAnalyzer_model.pkl` |

---

## 🚀 ENTRENAR MODELOS (desde nueva terminal)

### Opción 1: Entrenar Individual (RECOMENDADO)

Abre una **nueva terminal** (sin cerrar el servidor) y ejecuta:

#### 1️⃣ Entrenar Performance Predictor
```powershell
cd D:\PLATAFORMA EDUCATIVA\supervisado
python training\train_performance_adapted.py
```

**Esperado:**
```
Cargando datos...
Entrenando modelo...
✅ Modelo entrenado y guardado en: trained_models\PerformancePredictor_model.pkl
Precisión: 0.85 (85%)
```

#### 2️⃣ Entrenar Career Recommender
```powershell
cd D:\PLATAFORMA EDUCATIVA\supervisado
python training\train_career_recommender.py
```

#### 3️⃣ Entrenar Trend Predictor
```powershell
cd D:\PLATAFORMA EDUCATIVA\supervisado
python training\train_trend_predictor.py
```

#### 4️⃣ Entrenar Progress Analyzer
```powershell
cd D:\PLATAFORMA EDUCATIVA\supervisado
python training\train_progress_analyzer.py
```

---

### Opción 2: Entrenar Todos Secuencialmente

```powershell
cd D:\PLATAFORMA EDUCATIVA\supervisado

# Entrenar todos
python training\train_performance_adapted.py && `
python training\train_career_recommender.py && `
python training\train_trend_predictor.py && `
python training\train_progress_analyzer.py

echo "✅ Todos los modelos entrenados!"
```

---

## ⚙️ Requisitos para Entrenar

1. **Base de datos con datos**
   - Asegúrate que PostgreSQL está corriendo
   - Tabla `estudiantes` con datos
   - Tabla `calificaciones` con historial

2. **Variables de entorno configuradas**
   ```env
   DATABASE_URL=postgresql://postgres:1234@127.0.0.1:5432/educativa
   ```

3. **Dependencias instaladas**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🔍 Verificar Entrenamientos

Después de entrenar, verifica que los archivos fueron creados:

```powershell
ls D:\PLATAFORMA EDUCATIVA\supervisado\trained_models\

# Deberías ver:
# PerformancePredictor_model.pkl
# CareerRecommender_model.pkl
# TrendPredictor_model.pkl
# ProgressAnalyzer_model.pkl
```

---

## ✅ Verificar Modelos Cargados en el Servidor

Una vez entrenados, reinicia el servidor o consulta el endpoint:

```powershell
curl http://localhost:8001/health
```

**Respuesta esperada:**
```json
{
    "status": "healthy",
    "models_loaded": {
        "risk": true,        // ✅ Performance
        "career": true,      // ✅ Career
        "trend": true,       // ✅ Trend
        "progress": true     // ✅ Progress
    },
    "timestamp": "2025-11-25T..."
}
```

---

## 🎯 Probar Predicciones (Una vez entrenados)

### Test 1: Predicción de Riesgo
```powershell
curl -X POST http://localhost:8001/predict/risk `
  -H "Content-Type: application/json" `
  -d '{"student_id": 1}'
```

### Test 2: Recomendación de Carreras
```powershell
curl -X POST http://localhost:8001/predict/career `
  -H "Content-Type: application/json" `
  -d '{"student_id": 1}'
```

### Test 3: Predicción de Tendencia
```powershell
curl -X POST http://localhost:8001/predict/trend `
  -H "Content-Type: application/json" `
  -d '{"student_id": 1}'
```

### Test 4: Proyección de Progreso
```powershell
curl -X POST http://localhost:8001/predict/progress `
  -H "Content-Type: application/json" `
  -d '{"student_id": 1}'
```

---

## 📊 Documentación Interactiva

Una vez que el servidor está corriendo con modelos:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

Usa estos para explorar todos los endpoints disponibles.

---

## 🐛 Troubleshooting

### Error: "No hay datos disponibles"
**Causa:** Base de datos vacía
**Solución:** Asegúrate que hay estudiantes y calificaciones en PostgreSQL

### Error: "Connection refused"
**Causa:** PostgreSQL no está corriendo
**Solución:** 
```powershell
# Inicia PostgreSQL
pg_ctl -D "C:\Program Files\PostgreSQL\data" start
```

### Modelo toma mucho tiempo
**Normal:** Entrenar puede tomar de 10 segundos a 2 minutos
**Si toma más:** Reduce el número de estudiantes en el script

---

## 📝 Pasos Resumidos

```
1. Terminal 1: python -m uvicorn api_server:app --port 8001  (servidor)
2. Terminal 2: python training/train_performance_adapted.py   (entrenar)
3. Terminal 2: python training/train_career_recommender.py    (entrenar)
4. Terminal 2: python training/train_trend_predictor.py       (entrenar)
5. Terminal 2: python training/train_progress_analyzer.py     (entrenar)
6. Verificar:  curl http://localhost:8001/health
```

---

**Última actualización:** 25 de Noviembre 2025
**Estado:** ✅ Listo para entrenar

