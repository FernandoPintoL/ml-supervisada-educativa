# 🔄 ARQUITECTURA Y PIPELINE - ML SUPERVISADO

**Versión:** 1.0
**Última actualización:** 04/12/2025

---

## 📊 DIAGRAMA GENERAL DEL PIPELINE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PLATAFORMA EDUCATIVA ML                          │
│                     (Modelos Supervisados)                          │
└─────────────────────────────────────────────────────────────────────┘

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║              CAPA 1: RECOLECCIÓN DE DATOS             ║
        ║                  (PostgreSQL BD)                      ║
        ╚═══════════════════════════════════════════════════════╝
                      ↓
    ┌─────────────────────────────────────────────────┐
    │ users (id, desempeño_promedio, asistencia...)  │
    │ trabajos (estudiante_id, titulo, estado)       │
    │ calificaciones (trabajo_id, puntaje)           │
    │ rendimiento_academico (estudiante_id, promedio)│
    └─────────────────────────────────────────────────┘

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║         CAPA 2: CARGA DE DATOS (DataLoader)          ║
        ║     data/data_loader.py → supervisado/train_...py    ║
        ╚═══════════════════════════════════════════════════════╝

         Query SQL → Pandas DataFrame (100 estudiantes)

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║      CAPA 3: PREPROCESAMIENTO (Data Cleaning)       ║
        ║    - Eliminar nulls                                  ║
        ║    - Normalización (opcional)                        ║
        ║    - Feature Engineering                             ║
        ╚═══════════════════════════════════════════════════════╝

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║        CAPA 4: DIVISIÓN TRAIN/TEST                  ║
        ║    - 80% Training (primeras 3 semanas)              ║
        ║    - 20% Test (última semana)                       ║
        ║    División TEMPORAL, no aleatoria                  ║
        ╚═══════════════════════════════════════════════════════╝

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║         CAPA 5: ENTRENAMIENTO DE MODELOS            ║
        ║     Random Forest (50 árboles, max_depth=10)        ║
        ║     - 4 modelos en paralelo:                        ║
        ║       * Performance Predictor                       ║
        ║       * Career Recommender                          ║
        ║       * Trend Predictor                             ║
        ║       * Progress Analyzer                           ║
        ╚═══════════════════════════════════════════════════════╝

               Duración: ~1 segundo

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║       CAPA 6: VALIDACIÓN & EVALUACIÓN               ║
        ║    - Predecir sobre TEST set (datos no vistos)      ║
        ║    - Calcular métricas (R², Accuracy, etc)         ║
        ║    - Comparar vs modelo anterior                    ║
        ║    - Validación temporal (reciente vs histórico)    ║
        ╚═══════════════════════════════════════════════════════╝

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║       CAPA 7: ALMACENAMIENTO DE MODELOS             ║
        ║    ./trained_models/                                ║
        ║    ├── PerformancePredictor_model.pkl              ║
        ║    ├── CareerRecommender_model.pkl                 ║
        ║    ├── TrendPredictor_model.pkl                    ║
        ║    ├── ProgressAnalyzer_model.pkl                  ║
        ║    └── training_log.json                           ║
        ╚═══════════════════════════════════════════════════════╝

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║         CAPA 8: SERVICIO DE PREDICCIÓN              ║
        ║      FastAPI Server (api_server.py:8001)           ║
        ║    GET /predict/risk → Predicción de riesgo        ║
        ║    GET /predict/career → Recomendación de carrera  ║
        ║    GET /predict/trend → Tendencia de desempeño     ║
        ╚═══════════════════════════════════════════════════════╝

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║         CAPA 9: INTEGRACIÓN CON LARAVEL            ║
        ║    Laravel → HTTP Request → Python API             ║
        ║    ReportesController::reportesRiesgo()            ║
        ║    AnalisisRiesgoController::porCurso()            ║
        ╚═══════════════════════════════════════════════════════╝

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║         CAPA 10: ALMACENAMIENTO DE PREDICCIONES    ║
        ║    prediccion_riesgo table                         ║
        ║    prediccion_carrera table                        ║
        ║    prediccion_tendencia table                      ║
        ║    prediccion_progreso table                       ║
        ╚═══════════════════════════════════════════════════════╝

                                ↓

        ╔═══════════════════════════════════════════════════════╗
        ║         CAPA 11: VISUALIZACIÓN AL USUARIO           ║
        ║    React Frontend (/analisis-riesgo/*, /reportes)  ║
        ║    - Dashboard con predicciones                    ║
        ║    - Gráficos de tendencias                        ║
        ║    - Recomendaciones personalizadas               ║
        ╚═══════════════════════════════════════════════════════╝
```

---

## 🔍 FLUJO DETALLADO - POR ETAPA

### ETAPA 1: RECOLECCIÓN (BD)
**Archivo:** `train_models_simple.py` línea 103-124

**SQL Query:**
```sql
SELECT
    u.id, u.desempeño_promedio, u.asistencia_porcentaje,
    u.participacion_porcentaje, u.tareas_completadas,
    u.tareas_pendientes, u.actividad_hoy,
    AVG(c.puntaje) as promedio_calificaciones,
    COUNT(c.id) as total_calificaciones
FROM users u
LEFT JOIN trabajos t ON u.id = t.estudiante_id
LEFT JOIN calificaciones c ON t.id = c.trabajo_id
WHERE u.tipo_usuario = 'estudiante'
GROUP BY u.id
LIMIT 100
```

**Resultado:** DataFrame con 100 estudiantes × 10 features

---

### ETAPA 2: PREPARACIÓN
**Archivo:** `train_models_simple.py` línea 154-167

```python
# Limpiar datos
df_clean = df[features + ['target']].dropna()

# Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Resultado: 80 muestras training, 20 test
```

---

### ETAPA 3: ENTRENAMIENTO
**Archivo:** `train_models_simple.py` línea 170-178

```python
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)  # Entrenar SOLO con training
```

⚠️ **CRÍTICO:** Model.fit() SOLO usa datos de training, NO de test.

---

### ETAPA 4: VALIDACIÓN
**Archivo:** `train_models_simple.py` línea 181-184

```python
y_pred = model.predict(X_test)  # Predecir sobre datos NO VISTOS
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)
```

📌 **Importancia:** Validación sobre TEST asegura que modelo generaliza.

---

### ETAPA 5: ALMACENAMIENTO
**Archivo:** `train_models_simple.py` línea 191-194

```python
model_path = MODELS_DIR / 'PerformancePredictor_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
```

**Resultado:** Archivo binario .pkl (~365 KB)

---

### ETAPA 6: PREDICCIÓN EN RUNTIME
**Archivo:** `api/prediction_service.py`

```python
# Cargar modelo guardado
with open('trained_models/PerformancePredictor_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Obtener features del estudiante (en tiempo real)
features = [85.0, 90.0, 75.0, 10, 2, 1]

# Predicción
prediccion = model.predict([features])  # Output: 78.5
confianza = model.predict_proba([features])  # Probabilidad
```

---

## ⏰ FLUJO TEMPORAL - CUÁNDO SE EJECUTA QUÉ

### Opción 1: ENTRENAMIENTO MANUAL (Actual)
```
Administrador ejecuta:
  $ php artisan ml:train

Laravel/backend:
  ↓
Python (supervisado/train_models_simple.py):
  ├─ Conectar a BD
  ├─ Cargar 100 estudiantes
  ├─ Entrenar 4 modelos
  ├─ Guardar .pkl files
  └─ Log: "Modelos entrenados exitosamente"

Resultado: Modelos actualizados (~1 seg)
```

**Problema:**
- ❌ Manual, olvidadizo
- ❌ No sabe cuándo hay nuevos datos
- ❌ No valida si nuevo modelo es mejor

---

### Opción 2: REENTRENAMIENTO AUTOMÁTICO (Recomendado)
```
CRON Job (cada semana):
  ├─ Verificar si hay >50 nuevos registros
  ├─ Si SÍ: entrenar modelo nuevo
  ├─ Validar nuevo vs antiguo
  ├─ Si mejora: usar nuevo
  ├─ Si empeora: mantener antiguo
  └─ Log: resultados en JSON
```

**Beneficio:**
- ✅ Automático
- ✅ Datos frescos
- ✅ Métrica de mejora

---

## 🛠️ CÓMO EJECUTAR

### 1. Entrenamiento Completo (Manual)
```bash
# Desde raíz del proyecto educativo
php artisan ml:train --all

# O desde supervisado/
cd supervisado/
python train_models_simple.py
```

**Salida esperada:**
```
✓ Datos cargados: 100 estudiantes
✓ Performance Predictor entrenado (R²: 0.97)
✓ Career Recommender entrenado (Accuracy: 1.0)
✓ Trend Predictor entrenado (Accuracy: 0.90)
✓ Progress Analyzer entrenado (R²: 0.91)
✓ Modelos guardados en ./trained_models/
```

---

### 2. Validación REAL
```bash
cd supervisado/
python validation_real_models.py
```

**Salida esperada:**
```
[VALIDACIÓN] Cargando datos históricos...
✓ Datos cargados: 100 estudiantes
[DIVISIÓN TEMPORAL]
  Training: 85 estudiantes (hasta 2025-11-27)
  Test: 15 estudiantes (últimos 7 días)
[VALIDACIÓN] Performance Predictor
  ✓ R² Score: 0.72 (realista)
  ✓ RMSE: 11.23
  ✓ MAE: 8.45
```

---

### 3. Servidor de Predicción
```bash
cd supervisado/
python -m uvicorn api_server:app --port 8001

# En otra terminal, test:
curl http://localhost:8001/predict/risk?student_id=1
```

---

## 🚨 PUNTOS DE QUIEBRE (Failure Points)

### 1. BD sin datos
```
❌ Error: No se encontraron datos de estudiantes
Causa: Tabla users vacía o sin estudiantes
Solución: Verificar BD, cargar datos de seed
```

### 2. Features incompletos
```
❌ Error: Sin datos limpios después del filtrado
Causa: Muchos NULLs, valores faltantes
Solución: Revisar query SQL, agregar valores por defecto
```

### 3. Modelo no cargado
```
❌ Error: FileNotFoundError - PerformancePredictor_model.pkl
Causa: Archivo .pkl no existe
Solución: Ejecutar training primero (python train_models_simple.py)
```

### 4. API Server no responde
```
❌ Error: Connection refused (port 8001)
Causa: Servidor ML no iniciado
Solución: python -m uvicorn api_server:app --port 8001
```

### 5. Laravel no conecta a Python
```
❌ Error: cURL error 7 - Failed to connect
Causa: ML Service en diferente puerto/host
Solución: Verificar ML_SERVICE_URL en .env (debe ser localhost:8001)
```

---

## 📈 MONITOREO DEL PIPELINE

### Qué Monitorear
1. **Calidad de datos**
   - ¿Cuántos estudiantes hay?
   - ¿Cuántos registros de calificaciones?
   - ¿Cuántos nulls?

2. **Desempeño de modelos**
   - R² Score (debe estar >0.65)
   - Accuracy (debe estar >0.65)
   - Comparar: nuevo vs anterior

3. **Tiempo de ejecución**
   - Training: debe ser <5 segundos
   - Predicción: debe ser <100ms

4. **Errores**
   - ¿Qué falló?
   - ¿En qué etapa?
   - ¿Con qué estudiante?

---

## 🔐 SEGURIDAD

### Datos Sensibles
- ✅ Restricción de acceso a BD (usuario/password en .env)
- ✅ Validación de token en API (/auth/sanctum_auth.py)
- ⚠️ Models .pkl no están encriptados (considerar en producción)

### Recomendaciones
- [ ] Agregar logging de acceso a BD
- [ ] Encriptar files .pkl en producción
- [ ] Rate limiting en API (/predict/*)
- [ ] Auditoría de cambios de modelo

---

**Documento creado para mayor claridad en presentaciones y debugging.**

