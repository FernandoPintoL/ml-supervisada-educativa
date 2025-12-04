# ✅ DEPLOYMENT CHECKLIST - ML SUPERVISADO

**Versión:** 1.0
**Última actualización:** 04/12/2025
**Status:** Guía de configuración y despliegue

---

## 🎯 OBJETIVO

Asegurar que el pipeline ML está completamente funcional antes de presentación o producción.

---

## ✅ CHECKLIST PRE-DEPLOYMENT

### FASE 1: VERIFICACIÓN AMBIENTAL

- [ ] **1.1 - Python instalado y virtualenv activado**
  ```bash
  python --version  # Debe ser 3.9+
  pip list | grep scikit-learn  # Debe estar instalado
  ```
  📌 Archivos relevantes: `supervisado/.env`, `supervisado/requirements.txt`

- [ ] **1.2 - Variables de entorno configuradas**
  ```bash
  cat supervisado/.env | grep DB_
  ```
  Debe mostrar:
  ```
  DB_HOST=localhost
  DB_PORT=5432
  DB_DATABASE=educativa
  DB_USERNAME=postgres
  DB_PASSWORD=1234
  ```
  📌 Archivo: `supervisado/.env`

- [ ] **1.3 - PostgreSQL conectado**
  ```bash
  psql -h localhost -U postgres -d educativa -c "SELECT COUNT(*) FROM users WHERE tipo_usuario='estudiante';"
  ```
  Debe retornar número >0
  📌 Verificar: usuarios creados en `plataforma-educativa/database/`

- [ ] **1.4 - Dependencias Python instaladas**
  ```bash
  cd supervisado/
  pip install -r requirements.txt
  ```
  Debe completar sin errores
  📌 Archivo: `supervisado/requirements.txt`

---

### FASE 2: ENTRENAMIENTO INICIAL

- [ ] **2.1 - Ejecutar entrenamiento**
  ```bash
  cd supervisado/
  python train_models_simple.py
  ```
  ✓ Debe completar en <5 segundos
  ✓ Debe mostrar 4 modelos entrenados

  📌 Archivos creados:
  - `supervisado/trained_models/PerformancePredictor_model.pkl`
  - `supervisado/trained_models/CareerRecommender_model.pkl`
  - `supervisado/trained_models/TrendPredictor_model.pkl`
  - `supervisado/trained_models/ProgressAnalyzer_model.pkl`
  - `supervisado/trained_models/training_log.json`

- [ ] **2.2 - Verificar archivos .pkl existen**
  ```bash
  ls -lh supervisado/trained_models/
  ```
  Debe mostrar 4 archivos .pkl (50KB-400KB cada uno)

- [ ] **2.3 - Revisar training_log.json**
  ```bash
  cat supervisado/trained_models/training_log.json | python -m json.tool
  ```
  Debe mostrar métricas de entrenamiento

---

### FASE 3: VALIDACIÓN REAL (NUEVO - OPCIÓN A)

- [ ] **3.1 - Ejecutar validación con datos reales**
  ```bash
  cd supervisado/
  python validation_real_models.py
  ```
  ✓ Debe completar en 10-30 segundos
  ✓ Debe generar reporte en `supervisado/validation_reports/`

  📌 Archivo: `supervisado/validation_real_models.py` (nuevo)

- [ ] **3.2 - Revisar métricas honestas**
  ```bash
  cat supervisado/validation_reports/validation_report_*.json | python -m json.tool
  ```

  Buscar:
  ```json
  {
    "Performance Predictor": {
      "R² Score": 0.65-0.80  ← Más realista que 0.97
      "RMSE": 8-15,
      "MAE": 6-12
    },
    "Career Recommender": {
      "Accuracy": 0.65-0.78  ← Más realista que 1.0
      "Precision": 0.70-0.85,
      "Recall": 0.60-0.80,
      "F1-Score": 0.65-0.80
    }
  }
  ```

- [ ] **3.3 - Si métricas son BAJAS**
  ```
  Si R² < 0.50:
    → Problema: Features no son significativas
    → Solución: Revisar data_loader.py, ajustar features

  Si Accuracy < 0.55:
    → Problema: Modelo no mejor que random
    → Solución: Más datos, diferentes features
  ```

- [ ] **3.4 - Si métricas son BUENAS**
  ```
  Si R² > 0.65 y Accuracy > 0.70:
    ✓ Pasar a siguiente fase
    ✓ Documentar en presentación
  ```

---

### FASE 4: API SERVER (PYTHON)

- [ ] **4.1 - Iniciar servidor ML**
  ```bash
  cd supervisado/
  python -m uvicorn api_server:app --port 8001 --reload
  ```

  ✓ Debe mostrar:
  ```
  Uvicorn running on http://127.0.0.1:8001
  ```

  📌 Archivo: `supervisado/api_server.py`

- [ ] **4.2 - Test manual del servidor**
  En otra terminal:
  ```bash
  curl http://127.0.0.1:8001/predict/risk?student_id=1
  ```

  ✓ Debe retornar JSON con predicción:
  ```json
  {
    "student_id": 1,
    "risk_score": 0.75,
    "confidence": 0.82,
    "prediction": "alto"
  }
  ```

- [ ] **4.3 - Verificar endpoint /docs**
  Abrir en navegador:
  ```
  http://127.0.0.1:8001/docs
  ```

  ✓ Debe mostrar interfaz Swagger con endpoints disponibles

- [ ] **4.4 - Test con múltiples estudiantes**
  ```bash
  for i in {1..5}; do
    curl http://127.0.0.1:8001/predict/risk?student_id=$i
  done
  ```

  ✓ Todas las llamadas deben responder correctamente
  ✗ Si alguna falla: revisar logs en terminal API

---

### FASE 5: INTEGRACIÓN LARAVEL

- [ ] **5.1 - Verificar ML_SERVICE_URL en Laravel**
  ```bash
  cd plataforma-educativa/
  grep ML_SERVICE_URL .env
  ```

  Debe mostrar:
  ```
  ML_SERVICE_URL=http://localhost:8001
  ```

  📌 Archivo: `plataforma-educativa/.env`

- [ ] **5.2 - Ejecutar comando de entrenamiento desde Laravel**
  ```bash
  php artisan ml:train
  ```

  ✓ Debe completar sin errores
  ✓ Debe mostrar modelos entrenados

  📌 Archivo: `plataforma-educativa/app/Console/Commands/TrainML.php`

- [ ] **5.3 - Verificar que ReportesController llama API**
  ```bash
  grep -n "ML_SERVICE_URL" plataforma-educativa/app/Http/Controllers/ReportesController.php
  ```

  Debe mostrar referencias a la URL del servicio ML

  📌 Archivo: `plataforma-educativa/app/Http/Controllers/ReportesController.php` línea 269+

- [ ] **5.4 - Test de predicción en vivo**
  Navegar a:
  ```
  http://127.0.0.1:8000/reportes/riesgo
  ```

  ✓ Debe cargar tabla con predicciones
  ✓ Columnas: Estudiante, Score, Tipo Riesgo, Razón, Confianza, Fecha

---

### FASE 6: BD - PREDICCIONES GUARDADAS

- [ ] **6.1 - Verificar que predicciones se guardan en BD**
  ```bash
  psql -h localhost -U postgres -d educativa -c "SELECT COUNT(*) FROM prediccion_riesgo;"
  ```

  Debe retornar número >0 después de ejecutar `/reportes/riesgo`

  📌 Tabla: `prediccion_riesgo`

- [ ] **6.2 - Revisar estructura de predicciones**
  ```bash
  psql -h localhost -U postgres -d educativa -c "
    SELECT estudiante_id, score_riesgo, nivel_riesgo, fecha_prediccion
    FROM prediccion_riesgo LIMIT 3;
  "
  ```

  Debe mostrar predicciones con estructura correcta

- [ ] **6.3 - Verificar predicciones asociadas a estudiantes**
  ```bash
  psql -h localhost -U postgres -d educativa -c "
    SELECT u.nombre_completo, pr.score_riesgo
    FROM users u
    JOIN prediccion_riesgo pr ON u.id = pr.estudiante_id
    LIMIT 5;
  "
  ```

  ✓ Debe mostrar nombre + predicción correctamente asociados

---

### FASE 7: FRONTEND - VISUALIZACIÓN

- [ ] **7.1 - Verificar que reportes muestran datos**
  ```
  Navegar a: http://127.0.0.1:8000/reportes/riesgo
  ```

  ✓ Debe mostrar tabla con top 10 estudiantes en riesgo
  ✓ Columnas visibles: Estudiante, Score, Tipo Riesgo, Razón, Confianza, Fecha
  ✓ No debe haber errores en consola (F12 → Console)

- [ ] **7.2 - Verificar análisis por cursos**
  ```
  Navegar a: http://127.0.0.1:8000/analisis-riesgo/cursos
  ```

  ✓ Debe cargar lista de cursos
  ✓ Debe permitir seleccionar un curso
  ✓ Debe mostrar análisis del curso (métricas, estudiantes en riesgo)

- [ ] **7.3 - Verificar análisis de tendencias**
  ```
  Navegar a: http://127.0.0.1:8000/analisis-riesgo/tendencias
  ```

  ✓ Debe mostrar gráficos de tendencia
  ✓ Debe mostrar distribución por tendencia (mejorando, estable, declinando)

- [ ] **7.4 - No debe haber errores 403/500**
  Abrir Inspector (F12 → Network)

  ✗ Buscar requests rojas (errores 4xx/5xx)
  ✓ Todas las requests deben ser verdes (200)

---

### FASE 8: DOCUMENTACIÓN (OPCIÓN B)

- [ ] **8.1 - Verificar documentación técnica existe**
  ```bash
  ls -lh supervisado/*.md
  ```

  Debe mostrar:
  - `MODEL_DOCUMENTATION.md` ← Nuevo
  - `PIPELINE.md` ← Nuevo
  - `DEPLOYMENT_CHECKLIST.md` ← Este archivo

  📌 Archivos: `supervisado/MODEL_DOCUMENTATION.md`, `PIPELINE.md`

- [ ] **8.2 - Revisar que documentación es clara**
  ```bash
  head -50 supervisado/MODEL_DOCUMENTATION.md
  head -50 supervisado/PIPELINE.md
  ```

  ✓ Debe ser comprensible para no-técnicos
  ✓ Debe explicar problemas que resuelve

---

### FASE 9: PERFORMANCE & ESCALABILIDAD

- [ ] **9.1 - Test de carga**
  ```bash
  # En supervisado/, crear script de test:
  ab -n 100 -c 10 http://127.0.0.1:8001/predict/risk?student_id=1
  ```

  ✓ Debe completar sin errores
  ✓ Tiempo promedio: <100ms por request
  ✗ Si demora más: revisar BD, queries lentas

- [ ] **9.2 - Monitoreo de memoria**
  ```bash
  # Mientras servidor corre, monitorear:
  top | grep python
  ```

  ✓ Debe usar <200MB RAM
  ✗ Si usa más: posible memory leak

- [ ] **9.3 - Logging funciona**
  Buscar en output del servidor:
  ```
  [INFO] Predicción para student_id=1
  [INFO] Result: risk_score=0.75
  ```

  ✓ Debe haber logs informativos
  ✗ Si hay [ERROR]: revisar problema

---

## 🚀 PASO A PRODUCCIÓN

### Pre-Producción (AWS/Railway/Docker)

- [ ] **1. Build Docker**
  ```bash
  docker build -t ml-supervisado .
  docker run -p 8001:8001 ml-supervisado
  ```

- [ ] **2. Test en producción**
  ```bash
  curl https://ml-service.railway.app/predict/risk?student_id=1
  ```

- [ ] **3. Monitoreo 24/7**
  - Health check endpoint: `/health`
  - Logs a CloudWatch/DataDog
  - Alertas si endpoint falla

---

## 📋 CHECKLIST FINAL PARA PRESENTACIÓN

- [ ] **Modelos entrenados** (4/4)
- [ ] **Validación real ejecutada** (métricas honestas)
- [ ] **API server respondiendo** (localhost:8001)
- [ ] **Laravel conectado a API** (reportes cargando)
- [ ] **BD guardando predicciones** (tabla con datos)
- [ ] **Frontend mostrando datos** (sin errores 403/500)
- [ ] **Documentación completa** (3 archivos .md)
- [ ] **Performance OK** (<100ms por predicción)

---

## 🚨 TROUBLESHOOTING RÁPIDO

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named sklearn` | `pip install scikit-learn` |
| `Connection refused (port 8001)` | `python -m uvicorn api_server:app --port 8001` |
| `CORS error` | Verificar `api_server.py` tiene CORS habilitado |
| `BD connection error` | Verificar credenciales en `.env` |
| `Metrics very high (1.0, 0.97)` | Ejecutar `validation_real_models.py` |
| `Empty dataframe` | Verificar BD tiene estudiantes (tipo_usuario='estudiante') |
| `Model file not found` | Ejecutar `train_models_simple.py` primero |
| `Slow predictions (>1 sec)` | Revisar queries SQL, agregar índices en BD |

---

**Checklist creado para garantizar deployment exitoso.**

