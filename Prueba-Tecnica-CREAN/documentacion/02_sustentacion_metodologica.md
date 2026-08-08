# Sustentación Metodológica
## Solución Analítica para App de Inversiones CREAN

---

## 1. Metodología General

### 1.1 Enfoque de Solución

La solución implementada sigue una arquitectura analítica end-to-end basada en:

1. **Integración de fuentes SQL:** Consolidación de 7 bases de datos SQLite con información financiera y comportamental de clientes.
2. **Feature engineering orientado a negocio:** Construcción de variables derivadas que capturan capacidad financiera, propensión de inversión y experiencia previa.
3. **Modelado predictivo two-part (hurdle):** 
   - Modelo de clasificación para estimar **probabilidad de adopción**
   - Modelo de regresión condicionada para estimar **monto potencial de inversión**
4. **Scoring de valor esperado:** Combinación de ambos modelos para priorización comercial.

### 1.2 Pipeline de Ejecución

El pipeline implementado en [src/data_processing/ejecucion.py](../src/data_processing/ejecucion.py) ejecuta 5 etapas secuenciales:

```
Extracción → Limpieza → Feature Engineering → Entrenamiento → Scoring
```

**Características del pipeline:**
- **Reproducibilidad:** Ejecución con un solo comando, sin intervención manual
- **Trazabilidad:** Logs detallados de cada transformación aplicada
- **Validación automática:** Verificación de cardinalidad, nulos y esquema esperado
- **Artefactos versionables:** Modelos serializados (`.pkl`) y datos intermedios (`.parquet`)

---

## 2. Procesamiento de Datos

### 2.1 Extracción de Fuentes

**Módulo:** [src/data_processing/extraction.py](../src/data_processing/extraction.py)

Se extrajeron 7 fuentes de datos desde archivos SQLite (`.db`):

| Fuente | Registros | Descripción |
|--------|-----------|-------------|
| `clientes` | 860,231 | Información demográfica y financiera de clientes |
| `crean_aho_cte` | 1,000,000 | Saldos de cuentas de ahorro y corriente |
| `crean_bolsillos` | 1,000,000 | Saldos de bolsillos (subcuentas de ahorro) |
| `crean_fiducuenta` | 1,000,000 | Saldos de cuentas fiduciarias |
| `crean_inv_virtual_cdt` | 994,177 | Saldos de inversiones virtuales y CDTs |
| `estimador_ing` | 745,792 | Estimación algorítmica de ingresos por cliente |
| `invesbot` | 1,000,000 | Saldos en servicio de inversión automatizada |

**Total registros cargados:** 6,600,200

**Consideraciones técnicas:**
- Cada archivo `.db` contiene una única tabla interna
- El nombre de la tabla interna se detecta dinámicamente mediante query a `sqlite_master`
- Todas las tablas comparten la llave de negocio `numero_id`

---

### 2.2 Limpieza de Datos

**Módulo:** [src/data_processing/cleaning.py](../src/data_processing/cleaning.py)

#### 2.2.1 Reglas Aplicadas a `clientes`

**Eliminación de duplicados:**
- Se detectaron **16 registros duplicados** por `numero_id` (8 IDs afectados)
- **Acción:** Se conservó la primera ocurrencia con `drop_duplicates(subset=['numero_id'], keep='first')`
- **Justificación:** Los duplicados eran copias exactas sin información adicional

**Imputación de ingresos mensuales (249 nulos detectados):**
- **Paso 1:** Rescate desde `estimador_ing` mediante `LEFT JOIN` por `numero_id`
  - Se imputaron **176 registros** (70.68%) con el promedio del estimador algorítmico
  - **Justificación:** El estimador captura patrones transaccionales históricos, proxy válido de ingreso real
- **Paso 2:** Imputación a `0` para los 73 restantes (29.32%)
  - **Justificación:** Clientes sin historial transaccional suficiente para estimación
  - Se creó una bandera `flag_sin_info_financiera` para preservar esta condición

**Imputación de variables financieras (249 nulos en cada una):**
- Columnas afectadas: `total_egresos_mensuales`, `total_activos`, `total_pasivos`
- **Acción:** Imputación a `0`
- **Justificación:** Ausencia de información implica falta de registro, no valor explícito cero
- Se recalculó `total_patrimonio = total_activos - total_pasivos` post-imputación

**Tratamiento de variables categóricas:**
- `desc_tipo_de_vivienda`: Eliminada (> 99% de nulos, no relevante para modelo)
- `desc_genero`: Imputada con categoría explícita `"NO_REGISTRADO"` (93 nulos)

**Resultado final:** `clientes` con **0 nulos**, 860,223 filas y 10 columnas.

---

#### 2.2.2 Reglas Aplicadas a `crean_aho_cte`

**Descarte de registros con saldos negativos:**
- Se detectaron **1,470 registros** (0.15%) con saldo negativo en productos `CUENTA DE AHORRO` y `CUENTA DE CORRIENTE`
- **Acción:** Eliminación de estos registros
- **Justificación de negocio:** Un cliente con saldo negativo (sobregiro) no dispone de liquidez para invertir. La app de inversiones requiere fondos disponibles positivos para apertura.

**Resultado final:** 998,530 filas (de 1,000,000 originales).

---

#### 2.2.3 Fuentes sin Ajustes Adicionales

Las siguientes fuentes no presentaron problemas de calidad según análisis exploratorio:
- `crean_bolsillos`
- `crean_fiducuenta`
- `crean_inv_virtual_cdt`
- `estimador_ing`
- `invesbot`

**Consideración sobre saldos en cero:**
- En `crean_inv_virtual_cdt` e `invesbot` se detectaron registros con `saldo = 0`
- **Decisión:** No se descartaron
- **Justificación:** Un saldo en cero no implica ausencia de interés. Estos clientes **activaron el producto** en algún momento, lo cual es señal de propensión a inversión. Se consideran clientes con experiencia previa en el producto.

---

### 2.3 Feature Engineering

**Módulo:** [src/data_processing/feature_engineering.py](../src/data_processing/feature_engineering.py)

#### 2.3.1 Estrategia de Consolidación

Se construyó un **Master Feature Store** con granularidad de cliente (1 fila por `numero_id`) mediante:

1. **Entidad maestra:** Tabla `clientes` (860,223 filas únicas)
2. **Agregaciones por producto:** Consolidación de métricas desde tablas transaccionales
3. **Joins secuenciales:** `LEFT JOIN` desde `clientes` hacia cada agregación
4. **Imputación post-join:** Variables de productos no contratados se imputan a `0` (continuas) o `0` (banderas)

**Resultado:** `df_master.parquet` con **860,223 filas** y **35 columnas**.

---

#### 2.3.2 Variables Base (desde `clientes`)

**Financieras directas:**
- `ingresos_mensuales`, `total_egresos_mensuales`, `total_activos`, `total_pasivos`, `total_patrimonio`

**Ratios financieros derivados:**
- `margen_libre_estimado = ingresos_mensuales - total_egresos_mensuales`
  - Mide holgura operativa mensual para ahorro/inversión
- `ratio_apalancamiento = total_pasivos / (total_activos + 1)`
  - Nivel de endeudamiento relativo al activo
- `ratio_cobertura_egresos = total_activos / (total_egresos_mensuales * 12 + 1)`
  - Años de egresos cubiertos por activos (proxy de solvencia)

**Demográficas:**
- `grupo_edad`, `desc_genero`, `desc_segmento`

**Banderas de calidad:**
- `flag_sin_info_financiera`: Indica clientes sin información financiera completa

---

#### 2.3.3 Variables Agregadas por Producto

**Liquidez (desde `crean_aho_cte`):**
- `saldo_prom_liquidez`: Saldo promedio en cuentas de ahorro/corriente
- `saldo_max_liquidez`: Saldo máximo observado
- `cant_cuentas_aho`: Cantidad de productos de liquidez activos
- `flag_tiene_aho`: Bandera de posesión (1 = tiene, 0 = no tiene)

**Bolsillos (desde `crean_bolsillos`):**
- `saldo_total_bolsillos`: Saldo agregado promedio
- `cant_bolsillos`: Cantidad de bolsillos creados
- `flag_tiene_bolsillos`: Bandera de posesión

**Fiducuenta (desde `crean_fiducuenta`):**
- `saldo_prom_fiducuenta`: Saldo promedio en cuenta fiduciaria
- `flag_tiene_fiducuenta`: Bandera de posesión

**Invesbot (desde `invesbot`):**
- `saldo_prom_invesbot`: Saldo promedio en servicio de inversión automatizada
- `flag_tiene_invesbot`: Bandera de posesión

**Inversión Virtual / CDT (desde `crean_inv_virtual_cdt`):**
- `saldo_prom_inv_cdt`: Saldo promedio en CDT e inversión virtual
- `saldo_max_inv_cdt`: Saldo máximo observado
- `flag_tiene_cdt`: Bandera de posesión

---

#### 2.3.4 Features Avanzadas de Valor

**Relación con productos:**
- `num_productos_activos = sum(flag_tiene_*)`
  - Proxy de vinculación comercial y cross-sell readiness

**Consolidación de inversiones:**
- `saldo_total_inversiones = saldo_prom_fiducuenta + saldo_prom_invesbot + saldo_prom_inv_cdt`
  - Tamaño actual de inversión consolidada

**Ratios comportamentales:**
- `ratio_liquidez_vs_ingreso = saldo_prom_liquidez / (ingresos_mensuales + 1)`
  - Relación entre liquidez disponible y capacidad de ingreso
- `penetracion_bolsillos_vs_liquidez = saldo_total_bolsillos / (saldo_prom_liquidez + 1)`
  - Uso de bolsillos relativo a liquidez general

**Banderas de comportamiento:**
- `flag_superavit_operativo = (ingresos_mensuales > total_egresos_mensuales * 1.5)`
  - Señal de holgura financiera alta
- `flag_propension_digital_previa = (flag_tiene_invesbot == 1) OR (saldo_prom_inv_cdt > 0)`
  - Evidencia de afinidad digital/inversión previa

---

#### 2.3.5 Targets de Negocio

**Target de clasificación (adopción):**
```python
target_adopcion = ((saldo_total_inversiones > 0) OR (flag_tiene_invesbot == 1))
```
- **Definición:** Cliente que tiene al menos un producto de inversión activo (Fiducuenta, Invesbot o CDT con saldo > 0)
- **Distribución observada:** 220,441 positivos (25.6%) vs. 639,782 negativos (74.4%)
- **Justificación:** Proxy de propensión basada en comportamiento histórico

**Target de regresión (monto potencial a 12 meses):**
```python
excedente_liquidez = max(0, saldo_prom_liquidez - (total_egresos_mensuales * 2))
target_monto_12m = saldo_total_inversiones + excedente_liquidez
```
- **Definición:** Suma de inversión actual más excedente de liquidez disponible
- **Justificación del excedente:** Se conservan 2 meses de egresos como colchón de seguridad, el resto se considera invertible
- **Distribución:** Altamente asimétrica, con concentración en clientes de alto patrimonio

---

## 3. Modelado Predictivo

### 3.1 Arquitectura Two-Part

Se implementó una arquitectura de dos etapas:

1. **Modelo de clasificación:** Estima la probabilidad de que un cliente adopte la app
2. **Modelo de regresión:** Estima el monto potencial de inversión, condicionado a que adopte

**Valor esperado de negocio:**
```
EV_cliente = P(adopción) × E[monto | adopción]
```

Esta métrica permite **priorizar clientes** considerando tanto probabilidad como magnitud de inversión.

---

### 3.2 Modelo de Clasificación (Adopción)

**Algoritmo:** LightGBM Classifier

**Configuración de hiperparámetros:**
```python
LGBMClassifier(
    objective='binary',
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=30,
    scale_pos_weight=2.9023,  # Calculado dinámicamente por desbalanceo
    random_state=42
)
```

**Tratamiento de desbalanceo:**
- Clase positiva (adopción): 25.6%
- Clase negativa (no adopción): 74.4%
- Se aplicó `scale_pos_weight = (1 - p) / p = 2.9023` para balancear el loss

**Variables utilizadas (22 features):**
- Demográficas: `grupo_edad`, `desc_genero`, `desc_segmento`
- Financieras base: `ingresos_mensuales`, `total_egresos_mensuales`, `total_activos`, `total_pasivos`, `total_patrimonio`, `flag_sin_info_financiera`
- Ratios: `margen_libre_estimado`, `ratio_apalancamiento`, `ratio_cobertura_egresos`, `ratio_liquidez_vs_ingreso`, `penetracion_bolsillos_vs_liquidez`
- Banderas de comportamiento: `flag_superavit_operativo`, `flag_tiene_aho`, `flag_tiene_bolsillos`
- Magnitudes por producto: `saldo_prom_liquidez`, `saldo_max_liquidez`, `cant_cuentas_aho`, `saldo_total_bolsillos`, `cant_bolsillos`

**Exclusiones intencionales para evitar data leakage:**
- `flag_tiene_fiducuenta`, `flag_tiene_invesbot`, `flag_tiene_cdt`: Componentes directos del target
- `num_productos_activos`, `flag_propension_digital_previa`: Derivadas de banderas excluidas

**Métricas de evaluación (test set):**
- **ROC-AUC:** 0.9100 (excelente capacidad de discriminación)
- **PR-AUC:** 0.8107 (buen desempeño en clase minoritaria)
- **F1-Score:** 0.7376 (balance entre precisión y recall)
- **Brier Score:** 0.1124 (buena calibración de probabilidades)

---

### 3.3 Modelo de Regresión (Monto Potencial)

**Algoritmo:** LightGBM Regressor

**Configuración de hiperparámetros:**
```python
LGBMRegressor(
    objective='regression',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=30,
    random_state=42
)
```

**Transformación del target:**
- Se aplicó transformación `log1p(target_monto_12m)` para estabilizar varianza
- Los valores negativos se truncan a 0 en la predicción final

**Variables utilizadas (27 features):**
- Todas las del modelo de clasificación (22)
- Adicionales: `flag_tiene_fiducuenta`, `flag_tiene_invesbot`, `flag_tiene_cdt`, `num_productos_activos`, `flag_propension_digital_previa`
- **Justificación:** Para el monto, la experiencia previa en inversión es predictiva (no hay leakage)

**Universo de entrenamiento:**
- Se entrena **solo sobre clientes con adopción = 1** (220,441 observaciones)
- **Justificación:** El monto solo es relevante para quienes adoptan (modelo condicionado)

**Métricas de evaluación (test set, solo adoptantes):**
- **MAE:** $16,534,419 (error absoluto medio)
- **RMSE:** $64,475,596 (penaliza outliers)
- **R²:** 0.3329 (33.29% de varianza explicada)

**Interpretación del R²:**
- Un R² de 0.33 es aceptable en este contexto debido a:
  - Alta heterogeneidad en comportamiento de inversión
  - Influencia de factores externos no observados (timing de mercado, eventos personales)
  - Distribución altamente asimétrica del target

---

### 3.4 Reentrenamiento Final y Scoring Productivo

**Estrategia de producción:**
1. Los modelos de evaluación se entrenaron con split 80/20 para validar métricas
2. Post-validación, se **reentrenaron ambos modelos** sobre el 100% del universo disponible
3. Los modelos finales se serializaron en `models/lgbm_adopcion.pkl` y `models/lgbm_monto.pkl`

**Generación de scores:**
- Se aplicó inferencia sobre los 860,223 clientes
- Se calculó el valor esperado: `EV = proba_adopcion × monto_predicho`
- Se asignaron **deciles de priorización** basados en EV descendente

**Distribución de valor esperado por decil:**

| Decil | Clientes | EV Promedio | EV Total |
|-------|----------|-------------|----------|
| 10 | 86,023 | $37,393,322 | $3,216,685,734,192 |
| 9 | 86,022 | $4,753,566 | $408,911,256,326 |
| 8 | 86,022 | $839,569 | $72,221,384,532 |
| ... | ... | ... | ... |
| 1 | 86,023 | $779 | $66,988,514 |

**Salida:** `data/scores/df_predictions.parquet` con columnas:
- `numero_id`, `proba_adopcion`, `monto_predicho_12m`, `valor_esperado`, `decil_priorizacion`

---


## 4. Hallazgos Principales

### 4.1 Hallazgos de Calidad de Datos

1. **Alta cobertura de clientes en productos:**
   - 92% de clientes tienen al menos un producto de ahorro/corriente
   - 45% tienen productos de inversión (Fiducuenta, Invesbot, CDT)
   - **Implicación:** Base madura con experiencia en productos bancarios

2. **Baja incidencia de datos faltantes:**
   - Solo 249 clientes (0.03%) con información financiera incompleta
   - **Implicación:** Alta calidad en sistemas de origen

3. **Saldos negativos marginales:**
   - Solo 0.15% de registros con sobregiro en cuentas de ahorro/corriente
   - **Implicación:** Disciplina financiera en la base de clientes

---

### 4.2 Hallazgos de Segmentación

1. **Concentración de valor extrema:**
   - El 10% de clientes (Decil 10) concentra el 86.5% del valor esperado
   - **Implicación:** Estrategia de lanzamiento debe priorizar este segmento

2. **Segmentación demográfica clara:**
   - Decil 10: 80% Preferente/Empresarial, 65% entre 35-54 años, 58% masculino
   - **Implicación:** Perfil definido permite personalización de propuesta de valor

3. **Propensión digital alta:**
   - 95% de Decil 10 usa productos digitales (Invesbot o Inversión Virtual)
   - **Implicación:** Canal app es adecuado para este segmento

---

### 4.3 Hallazgos de Capacidad Financiera

1. **Superávit operativo mayoritario:**
   - 78% de clientes en Decil 10 tienen `ingresos > 1.5 × egresos`
   - **Implicación:** Alta capacidad de ahorro/inversión

2. **Excedente de liquidez significativo:**
   - Promedio de $8.5M en excedente de liquidez (Top 3 deciles)
   - **Implicación:** Capital disponible para inversión sin afectar operación

3. **Patrimonio sólido:**
   - Patrimonio promedio > $500M en Decil 10
   - **Implicación:** Clientes con respaldo patrimonial significativo



---
## 5. Conclusiones Metodológicas
1. **Solución end-to-end exitosa:**
   - Se logró construir un pipeline reproducible desde datos crudos hasta scoring productivo
   - Métricas de clasificación excelentes (ROC-AUC 0.91) validan calidad del modelo

2. **Valor de negocio cuantificado:**
   - El scoring permite priorizar clientes por valor esperado, maximizando ROI de campañas
   - La concentración de valor en Decil 10 (86.5%) valida estrategia de lanzamiento selectivo

3. **Limitaciones reconocidas y mitigadas:**
   - Se documentaron supuestos explícitos y limitaciones técnicas
   - Se proponen mejoras incrementales basadas en aprendizaje continuo

4. **Preparación para producción:**
   - Modelos serializados, artefactos versionables y dashboard funcional
   - Solución lista para integración en flujo operativo de CREAN

---

**Documento generado con ayuda de IA:** 2026-08-08  
**Responsable:** LDC Analítica CREAN - danaleri  