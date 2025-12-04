# 📚 DOCUMENTACIÓN TÉCNICA - MODELOS ML SUPERVISADOS

**Versión:** 1.0
**Última actualización:** 04/12/2025
**Status:** ✅ DOCUMENTACIÓN COMPLETA

---

## 📋 ÍNDICE

1. [Performance Predictor](#1-performance-predictor)
2. [Career Recommender](#2-career-recommender)
3. [Trend Predictor](#3-trend-predictor)
4. [Progress Analyzer](#4-progress-analyzer)
5. [Limitaciones Conocidas](#limitaciones-conocidas)
6. [Próximas Mejoras](#próximas-mejoras)

---

## 1. PERFORMANCE PREDICTOR

### Propósito
Predice las calificaciones promedio de estudiantes basado en métricas académicas actuales.

### Problema que Resuelve
Permite identificar qué tan bien está desempeñándose un estudiante actualmente y predecir su trayectoria académica futura.

### Algoritmo
**Random Forest Regressor** con parámetros:
```python
n_estimators=50      # 50 árboles
max_depth=10         # Profundidad máxima de 10 niveles
random_state=42      # Reproducibilidad
n_jobs=-1            # Usar todos los cores disponibles
```

**Por qué Random Forest?**
- Interpetable (puedes ver qué features importan más)
- Robusto a outliers
- No requiere normalización de datos
- Maneja relaciones no-lineales

### Features de Entrada (Características)
| Feature | Tipo | Rango | Descripción |
|---------|------|-------|-------------|
| `desempeño_promedio` | Float | 0-100 | Promedio académico del estudiante |
| `asistencia_porcentaje` | Float | 0-100 | % de asistencia a clases |
| `participacion_porcentaje` | Float | 0-100 | % de participación en clase |
| `tareas_completadas` | Int | 0-N | Número de tareas entregadas |
| `tareas_pendientes` | Int | 0-N | Número de tareas faltantes |
| `actividad_hoy` | Int | 0-1 | ¿Tuvo actividad académica hoy? (0/1) |

### Target (Predicción)
`promedio_calificaciones`: Promedio de todas las calificaciones obtenidas (0-100)

### Datos de Entrenamiento
```
Estudiantes: 100
Registros: ~200-250 (según cantidad de calificaciones)
Split: 80% training, 20% test
```

### Métrica de Evaluación
**R² Score (Coefficient of Determination)**
```
Interpretación:
- R² = 1.0  → Predicción perfecta
- R² = 0.8+ → Excelente (explica >80% de varianza)
- R² = 0.5-0.8 → Bueno (explica 50-80% de varianza)
- R² = 0.0  → Predicción no mejor que promedio
- R² < 0.0  → Predicción peor que promedio
```

**Otras Métricas:**
- **RMSE** (Root Mean Square Error): Error cuadrático promedio
- **MAE** (Mean Absolute Error): Error medio absoluto

### Validación Actual
Ejecutar: `python validation_real_models.py`

**Métricas esperadas (HONESTAS):**
- R²: 0.65-0.80 (no 0.97 como antes)
- RMSE: 8-15 puntos
- MAE: 6-12 puntos

### Casos de Uso
1. **Predicción de calificaciones finales**
   - Input actual del estudiante → Salida: predicción de nota final

2. **Identificación de estudiantes en riesgo**
   - Si predicción < 3.0 → Riesgo académico

3. **Monitoreo progresivo**
   - Tracking: predicción anterior vs predicción nueva

### Limitaciones
- ❌ No captura cambios súbitos de comportamiento
- ❌ Depende de calidad de datos de entrada
- ❌ Con <100 estudiantes, puede overficar
- ❌ No considera factores externos (enfermedad, problemas familiares)

### Cómo Llamarlo

```python
# En Python (training/development)
from models.performance_predictor import PerformancePredictor

predictor = PerformancePredictor()
predictor.load_model('trained_models/PerformancePredictor_model.pkl')

features = [85.0, 90.0, 75.0, 10, 2, 1]  # desempeño, asistencia, etc
prediccion = predictor.predict([features])
print(f"Calificación predicha: {prediccion[0]:.2f}")
```

```bash
# Vía API (desde Laravel)
POST /api/predict/performance
{
  "student_id": 123
}

Response:
{
  "prediccion": 78.5,
  "confianza": 0.82,
  "features_usados": [85.0, 90.0, 75.0, 10, 2, 1]
}
```

---

## 2. CAREER RECOMMENDER

### Propósito
Recomienda carreras universitarias basado en el desempeño académico del estudiante.

### Problema que Resuelve
Orientación vocacional data-driven: predecir qué carrera es mejor para cada estudiante.

### Algoritmo
**Random Forest Classifier** binario (2 clases):

```python
n_estimators=50
max_depth=10
random_state=42
```

**Clases:**
- Clase 0: "Bajo rendimiento" (desempeño ≤ 70)
- Clase 1: "Alto rendimiento" (desempeño > 70)

### Features de Entrada
| Feature | Tipo | Descripción |
|---------|------|-------------|
| `desempeño_promedio` | Float | Promedio académico |
| `asistencia_porcentaje` | Float | % de asistencia |
| `participacion_porcentaje` | Float | % de participación |
| `promedio_rendimiento` | Float | Promedio de todas las evaluaciones |

### Target
Clasificación binaria de rendimiento académico general.

### Métrica de Evaluación
**Accuracy** (Exactitud)
```
Interpretación:
- Accuracy = 0.85+ → Excelente (>85% de predicciones correctas)
- Accuracy = 0.70-0.85 → Bueno (70-85%)
- Accuracy = 0.50-0.70 → Aceptable (50-70%)
- Accuracy < 0.50 → Débil
```

**Adicionales:**
- **Precision**: De los que predijo "Alto", ¿cuántos realmente lo son?
- **Recall**: De los "Altos reales", ¿cuántos detectó?
- **F1-Score**: Balance entre Precision y Recall

### Validación Actual
Ejecutar: `python validation_real_models.py`

**Métricas esperadas (HONESTAS):**
- Accuracy: 0.65-0.78 (no 1.0 como antes)
- Precision: 0.70-0.85
- Recall: 0.60-0.80
- F1-Score: 0.65-0.80

### Casos de Uso
1. **Recomendación de carrera**
   - Si predicción = "Alto rendimiento" → Recomendar carreras exigentes
   - Si predicción = "Bajo rendimiento" → Recomendar carreras técnicas

2. **Orientación educativa**
   - Padres y docentes toman decisiones basadas en datos

### Limitaciones
- ❌ Simplificación binaria (realidad es más compleja)
- ❌ No considera intereses personales del estudiante
- ❌ No valida elección real de carrera después
- ❌ Carreras recomendadas no son dinámicas (hardcodeadas)

### Cómo Llamarlo

```python
from models.career_recommender import CareerRecommender

recommender = CareerRecommender()
prediccion_clase = recommender.predict([features])

if prediccion_clase == 1:
    print("Alto rendimiento → Carreras: Ingeniería, Medicina")
else:
    print("Bajo rendimiento → Carreras: Técnicas, Comercio")
```

---

## 3. TREND PREDICTOR

### Propósito
Predice la tendencia de desempeño del estudiante: mejorando, estable, declinando, fluctuando.

### Problema que Resuelve
Detectar cambios en el comportamiento académico para intervenir rápidamente.

### Algoritmo
**Random Forest Classifier** multiclase (4 clases):

```python
Clases:
- 0: "Mejorando" (desempeño ascendente)
- 1: "Estable" (desempeño sin cambios)
- 2: "Declinando" (desempeño descendente)
- 3: "Fluctuando" (cambios irregulares)
```

### Features de Entrada
| Feature | Descripción |
|---------|-------------|
| `asistencia_porcentaje` | % de asistencia |
| `participacion_porcentaje` | % de participación |
| `actividad_hoy` | Actividad en últimas 24h |

### Target
Clasificación de tendencia en 4 categorías.

### Métrica de Evaluación
**Accuracy** (Exactitud)
- Esperado: 0.70-0.85

### Limitaciones Críticas
- ❌ Tendencias muy recientes (últimos días) no son significativas
- ❌ Necesita histórico de MESES para validar cambios reales
- ❌ Sin causalidad: no sabe si cambio es por intervención o no

### ⚠️ ESTADO ACTUAL
**Funciona técnicamente pero NO validado en realidad.**
Necesita:
- Datos históricos de 3+ meses
- Validación comparando predicción vs resultado real

---

## 4. PROGRESS ANALYZER

### Propósito
Analiza el progreso académico del estudiante en el tiempo.

### Algoritmo
**Random Forest Regressor**

### Features
| Feature | Descripción |
|---------|-------------|
| `tareas_completadas` | Número de tareas entregadas |
| `tareas_pendientes` | Número de tareas faltantes |
| `actividad_hoy` | Actividad en últimas 24h |

### Target
Calificación de progreso (0-100)

### Métrica
**R² Score**
- Esperado: 0.70-0.85

### Limitaciones
- ❌ Ambiguo: ¿qué es "progreso"?
- ❌ No distingue entre progreso positivo y negativo
- ❌ Poco validado en realidad

---

## LIMITACIONES CONOCIDAS

### Todas los modelos
1. **Datos Pequeños (100 estudiantes)**
   - Riesgo de overfitting
   - Mejora con 500+ estudiantes

2. **Features Estáticas**
   - Métricas tomadas en un momento
   - No capturan evolución temporal

3. **Falta de Context**
   - No considera factores externos
   - Problemas familiares, salud, etc.

4. **Sin Validación Longitudinal**
   - Modelos NO validados contra "realidad futura"
   - Necesita: predicción hace 1 mes vs resultado hoy

5. **Desbalance de Clases**
   - Career Recommender: pocos estudiantes "Alto rendimiento"
   - Puede sesgar predicciones

### En Producción
- ML Service en localhost:8001 (no escalable)
- Sin caché de predicciones (lento con muchos estudiantes)
- Reentrenamiento manual (debería ser automático)

---

## PRÓXIMAS MEJORAS

### Corto Plazo (1-2 semanas)
- [ ] Ejecutar `validation_real_models.py` para métricas honestas
- [ ] Documentar exactamente por qué métricas son bajas
- [ ] Ajustar hiperparámetros si es necesario

### Mediano Plazo (1 mes)
- [ ] Recolectar 200+ estudiantes reales
- [ ] Reentrenamiento con más datos
- [ ] Validación longitudinal (predicción vs realidad después)

### Largo Plazo (2-3 meses)
- [ ] Agregar más features (comportamiento, social, etc.)
- [ ] Implementar reentrenamiento automático mensual
- [ ] Usar Deep Learning si justifica

---

**Documento creado para mayor transparencia y credibilidad en presentaciones.**

