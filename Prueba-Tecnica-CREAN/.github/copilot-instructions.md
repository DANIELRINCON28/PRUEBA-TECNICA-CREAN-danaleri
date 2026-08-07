# CLAUDE.md — Contexto del Proyecto: App de Inversiones CREAN

Este archivo es el contexto maestro del proyecto. Léelo por completo antes de trabajar en cualquier fase del `plan_ejecucion.md`. No asumas nada que contradiga lo aquí documentado.

## ⚠️ 0. Principio rector — esto simula datos de producción bancaria real

Este ejercicio se trata **como si un error de limpieza pudiera llegar a un comité de riesgo o a un reporte regulatorio.** Reglas obligatorias, sin excepción:

1. **Ningún dato "raro" se corrige por reflejo.** Antes de modificar, eliminar o transformar cualquier valor, primero busca la explicación de negocio.
2. **`numero_id` negativos son valores reales, no errores de captura.** No se eliminan, no se convierten a valor absoluto. Se documentan como hallazgo y, por defecto, se conservan tal cual (un identificador es una llave, no una magnitud — su signo no afecta su función de join).
3. **Los nulos NO se imputan por defecto.** Nunca rellenes con media/mediana/moda/cero automáticamente. Solo se puede reconstruir un valor si **otra fuente lo confirma explícitamente** (ej. `ingresos_mensuales` nulo en `clientes` pero presente en `estimador_ing`), y eso se documenta como "dato recuperado de fuente cruzada", no como imputación estadística. Si no hay fuente que lo confirme, el valor queda nulo explícito + flag `dato_no_disponible`, y su tratamiento se decide en la fase de modelado (nunca relleno silencioso).
4. **Toda regla de limpieza se registra en el "Log de Decisiones de Calidad de Datos"** (columna, problema, regla aplicada, justificación, fuente de soporte si aplica).
5. **Ante ambigüedad, detente y pregunta antes de aplicar una transformación irreversible** (drop de filas/columnas, cast de tipos con pérdida de información). No asumas "lo más común" sin dejarlo explícito y sin confirmación.

---

## 1. Objetivo de negocio

CREAN (Creación y Aceleración de Nuevos Negocios) lanzará una nueva App de inversiones para clientes actuales del banco. Necesitamos:

1. Identificar qué clientes tienen mayor probabilidad de adoptarla (próximos 12 meses).
2. Estimar cuánto dinero (monto) podrían invertir esos clientes.
3. Dimensionar la oportunidad de negocio total (AUM potencial, # clientes prioritarios).
4. Conectar los resultados con los procesos operativos de CREAN (ver sección 6).

El entregable final combina: modelo de datos, modelos analíticos, dimensionamiento de negocio, diagrama de procesos y un tablero.

---

## 2. Formato de las fuentes

**Las 7 tablas viven en un archivo `.db` (SQLite)**, no en CSV. Todo acceso a datos debe hacerse vía `sqlite3` o `sqlalchemy` con queries explícitas (`SELECT * FROM tabla`), nunca asumir `pd.read_csv`. Al iniciar cualquier fase que toque datos, primero valida la conexión y lista las tablas (`SELECT name FROM sqlite_master WHERE type='table';`) antes de asumir su contenido.

---

## 3. Diccionario de datos (fuente de verdad — no inventar columnas)

### `clientes` (dimensión, grano = 1 fila por cliente)
- `numero_id`
- `grupo_edad`
- `desc_genero`
- `desc_segmento`
- `desc_tipo_de_vivienda`
- `ingresos_mensuales`
- `total_egresos_mensuales`
- `total_activos`
- `total_pasivos`
- `total_patrimonio`

### `crean_aho_cte`, `crean_bolsillos`, `crean_fiducuenta`, `crean_inv_virtual_cdt`, `invesbot`
(todas comparten la misma estructura — son tablas de hechos, grano = cliente–producto–fecha)
- `fecha`
- `numero_id`
- `producto`
- `saldo`

### `estimador_ing` (grano = 1 fila por cliente, sin fecha)
- `numero_id`
- `producto`
- `estimador_ingreso`

---

## 4. Diagnóstico de datos ya validado (no recalcular desde cero, usar como referencia de control)

| Tabla | Filas | Clientes únicos | % Overlap con clientes |
|---|---|---|---|
| clientes | 860,231 | 860,223 (8 duplicados a limpiar) | 100.0% |
| crean_aho_cte | 1,000,000 | 475,719 | 55.3% |
| crean_bolsillos | 1,000,000 | 260,714 | 30.31% |
| crean_fiducuenta | 1,000,000 | 181,021 | 21.04% |
| crean_inv_virtual_cdt | 994,177 | 84,104 | 9.78% |
| estimador_ing | 745,792 | 745,792 | 86.7% |
| **invesbot** | 1,000,000 | **5,214** | **0.61%** |

Después de tu propio EDA, si estos números no coinciden aproximadamente, detente y repórtalo — puede indicar un error de carga o join.

---

## 5. Supuestos de negocio ya definidos (no redefinir sin avisar)

1. **`invesbot` es el proxy de adopción histórica** de la nueva App (target/label para el modelo de propensión). Es un servicio digital de inversión ya existente, análogo al nuevo producto.
2. El problema tiene **desbalance extremo** (~1 positivo por cada 165 negativos). Usar métricas robustas a desbalance (PR-AUC, no solo accuracy/ROC-AUC) y ajustar el modelo (`scale_pos_weight`, `class_weight='balanced'`).
3. El problema se resuelve como **modelo de dos partes**:
   - Modelo 1: P(adopción) — clasificación.
   - Modelo 2: E(monto | adopción=1) — regresión, entrenado solo sobre clientes con evidencia de inversión (invesbot + fiducuenta + inv_virtual_cdt).
   - Score de negocio final: `valor_esperado = P(adopción) × monto_esperado`.
4. **Anti-leakage obligatorio:** define un punto de corte temporal. Las features (comportamiento en aho_cte, bolsillos, fiducuenta, CDT) deben construirse SOLO con datos anteriores al corte. El target (aparición en invesbot) se valida SOLO con datos posteriores al corte. Nunca uses el saldo de invesbot del mismo periodo como feature para predecir invesbot.
5. Un cliente ausente en una tabla de producto (ej. no aparece en `crean_fiducuenta`) significa que **no tiene ese producto**, no es un dato faltante a imputar con la media. Rellenar con 0 / flag binario tras el left join desde `clientes`.

---

## 6. Convenciones técnicas

- Lenguaje: Python (pandas, scikit-learn, lightgbm/xgboost, matplotlib/plotly).
- Nombres de variables y comentarios: **español**.
- Antes de escribir código para una fase nueva, primero explica en texto el plan de esa fase (3-5 pasos) y espera confirmación si el paso es ambiguo o de alto impacto (ej. definición del punto de corte, tratamiento de outliers).
- Cada fase debe terminar con evidencia verificable (shape, métricas, gráfico) — no avances a la siguiente fase sin mostrarla.
- Los supuestos que tomes DURANTE el desarrollo (ej. cómo trataste un duplicado específico) deben quedar documentados en una celda markdown junto al código, no solo en el chat.
- **Control de créditos Copilot:** no cargues tablas completas (900K+ filas) en el contexto del chat. Trabaja siempre con `head()`, `info()`, `sample()` o queries agregadas ejecutadas por código, mostrando solo resúmenes. El presupuesto de créditos por fase está documentado en `plan_ejecucion.md` — revísalo antes de iniciar cada fase.

---

## 7. Procesos CREAN a los que debe conectar la solución

- Conciliar transacciones y contabilidad
- Gestionar ingresos y gastos
- Gestionar el uso del servicio
- Administrar información
- Monitorear el servicio
- Afiliar / Desafiliar al servicio
- Administrar el servicio

Cada entregable analítico (ABT, modelo, score, dashboard) debe poder mapearse explícitamente a uno o más de estos procesos — no lo dejes implícito.

---

## 8. Plan de ejecución de referencia

El detalle fase por fase está en `plan_ejecucion.md`. Trabaja una fase a la vez, en el orden ahí definido. No saltes fases ni las combines salvo instrucción explícita.