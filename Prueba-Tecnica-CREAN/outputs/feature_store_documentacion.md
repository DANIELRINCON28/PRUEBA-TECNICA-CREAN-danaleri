# Documentacion del Master Feature Store

## Objetivo
Este documento describe las variables del `df_master` para soportar:

- Modelo de clasificacion: probabilidad de adopcion de la App de inversiones.
- Modelo de regresion: monto potencial de inversion en los proximos 12 meses.

La entidad maestra es `clientes`, con 1 fila por `numero_id` y consolidacion mediante `LEFT JOIN`.

## Features creadas y justificacion

### 1) Features base del cliente (perfil financiero)

- `ingresos_mensuales`: capacidad de flujo de entrada.
- `total_egresos_mensuales`: presion de gasto.
- `total_activos`: respaldo patrimonial.
- `total_pasivos`: carga financiera.
- `total_patrimonio`: solvencia neta.
- `flag_sin_info_financiera`: indicador de calidad/completitud de informacion original.

### 2) Ratios financieros base (creadas en feature engineering)

- `margen_libre_estimado = ingresos_mensuales - total_egresos_mensuales`
  - Mide holgura operativa mensual y capacidad de ahorro.
- `ratio_apalancamiento = total_pasivos / (total_activos + 1)`
  - Mide exposicion a deuda relativa al activo.
- `ratio_cobertura_egresos = total_activos / (total_egresos_mensuales * 12 + 1)`
  - Aproxima cuantos anos de egresos estan cubiertos por activos.

### 3) Features agregadas por producto

- Liquidez (`crean_aho_cte`):
  - `saldo_prom_liquidez`
  - `saldo_max_liquidez`
  - `cant_cuentas_aho`
  - `flag_tiene_aho`
- Bolsillos (`crean_bolsillos`):
  - `saldo_total_bolsillos` (agregado por media segun requerimiento)
  - `cant_bolsillos`
  - `flag_tiene_bolsillos`
- Fiducuenta (`crean_fiducuenta`):
  - `saldo_prom_fiducuenta`
  - `flag_tiene_fiducuenta`
- Invesbot (`invesbot`):
  - `saldo_prom_invesbot`
  - `flag_tiene_invesbot`
- CDT virtual (`crean_inv_virtual_cdt`):
  - `saldo_prom_inv_cdt`
  - `saldo_max_inv_cdt`
  - `flag_tiene_cdt`

Justificacion: estas variables capturan profundidad de relacion financiera, volumen transaccional y experiencia previa en productos de ahorro/inversion.

### 4) Features avanzadas de valor (creadas en feature engineering)

- `num_productos_activos`
  - Proxy de vinculacion comercial y cross-sell readiness.
- `saldo_total_inversiones`
  - Tamaño actual de inversion consolidada.
- `ratio_liquidez_vs_ingreso`
  - Relacion entre liquidez disponible y capacidad de ingreso.
- `penetracion_bolsillos_vs_liquidez`
  - Uso de bolsillos relativo a liquidez general.
- `flag_superavit_operativo`
  - Señal de holgura financiera alta (ingresos > 1.5 * egresos).
- `flag_propension_digital_previa`
  - Evidencia de afinidad digital/inversion previa.

## Targets de modelado

### Target 1: clasificacion

- `target_adopcion = ((saldo_total_inversiones > 0) OR (flag_tiene_invesbot == 1))`
- Uso: entrenar modelo de propension de adopcion (0/1).

### Target 2: regresion

- `excedente_liquidez = max(0, saldo_prom_liquidez - (total_egresos_mensuales * 2))`
- `target_monto_12m = saldo_total_inversiones + excedente_liquidez`
- Uso: estimar monto potencial invertible a 12 meses.

## Variables recomendadas por modelo

### Modelo de clasificacion (propension de adopcion)

Variables recomendadas:

- Demograficas: `grupo_edad`, `desc_genero`, `desc_segmento`.
- Financieras base: `ingresos_mensuales`, `total_egresos_mensuales`, `total_activos`, `total_pasivos`, `total_patrimonio`.
- Ratios: `margen_libre_estimado`, `ratio_apalancamiento`, `ratio_cobertura_egresos`.
- Relacion con productos: todas las `flag_tiene_*`, `num_productos_activos`, `cant_cuentas_aho`, `cant_bolsillos`.
- Saldos por producto: `saldo_prom_liquidez`, `saldo_prom_fiducuenta`, `saldo_prom_invesbot`, `saldo_prom_inv_cdt`, `saldo_total_bolsillos`.
- Comportamentales derivadas: `ratio_liquidez_vs_ingreso`, `penetracion_bolsillos_vs_liquidez`, `flag_superavit_operativo`, `flag_propension_digital_previa`.

Variable objetivo:

- `target_adopcion`.

### Modelo de regresion (monto potencial 12M)

Variables recomendadas:

- Todas las anteriores excepto `target_adopcion`.
- Especial enfasis en magnitudes continuas: `saldo_*`, `margen_libre_estimado`, `ratio_*`, `num_productos_activos`.

Variable objetivo:

- `target_monto_12m`.

## Columnas a NO usar como features de entrenamiento

- Identificador tecnico: `numero_id`.
- Objetivos (depende del modelo):
  - Para clasificacion: no usar `target_adopcion` como feature.
  - Para regresion: no usar `target_monto_12m` como feature.
- Variables auxiliares derivadas de target en entrenamiento directo:
  - `excedente_liquidez` (si se materializa en version futura), para evitar fuga de informacion si el target se define con la misma formula.
- Campos de fecha a nivel transaccional (`fecha`) y `producto` original de tablas detalle:
  - No quedan en `df_master`; fueron consumidos en agregaciones.

## Reglas de calidad aplicadas

- Consolidacion estricta con `LEFT JOIN` desde `clientes`.
- Imputacion en `0.0` de variables continuas de productos no contratados.
- Imputacion en `0` de banderas de posesion (`flag_tiene_*`).
- Validacion final de cardinalidad esperada (860,223 filas) y nulos totales en `0`.
