"""
src/data_processing/cleaning.py
Modulo de limpieza de datos para el pipeline ETL de CREAN.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class DataCleaner:
    """Aplica reglas de limpieza de negocio sobre las fuentes extraidas."""

    REQUIRED_SOURCES = {
        "clientes",
        "crean_aho_cte",
        "crean_bolsillos",
        "crean_fiducuenta",
        "crean_inv_virtual_cdt",
        "estimador_ing",
        "invesbot",
    }

    def __init__(self, raw_dataframes: Dict[str, pd.DataFrame]):
        """Inicializa el limpiador con el diccionario retornado por DataExtractor."""
        missing_sources = self.REQUIRED_SOURCES.difference(raw_dataframes.keys())
        if missing_sources:
            raise KeyError(
                "Faltan fuentes requeridas para limpieza: "
                f"{sorted(missing_sources)}"
            )
        self.raw_dataframes = raw_dataframes

    def clean_all_sources(self) -> Dict[str, pd.DataFrame]:
        """Ejecuta la limpieza completa y retorna un nuevo diccionario de DataFrames."""
        logging.info("Iniciando proceso de limpieza de fuentes...")

        cleaned_dataframes: Dict[str, pd.DataFrame] = {}

        cleaned_dataframes["clientes"] = self._clean_clientes(
            clientes_df=self.raw_dataframes["clientes"],
            estimador_ing_df=self.raw_dataframes["estimador_ing"],
        )
        cleaned_dataframes["crean_aho_cte"] = self._clean_crean_aho_cte(
            self.raw_dataframes["crean_aho_cte"]
        )

        # En EDA no se definieron transformaciones adicionales para estas fuentes.
        passthrough_sources = [
            "crean_bolsillos",
            "crean_fiducuenta",
            "crean_inv_virtual_cdt",
            "estimador_ing",
            "invesbot",
        ]
        for source in passthrough_sources:
            cleaned_dataframes[source] = self.raw_dataframes[source].copy()
            logging.info(
                "Fuente '%s' sin ajustes adicionales segun EDA: %s filas.",
                source,
                f"{len(cleaned_dataframes[source]):,}",
            )

        logging.info("Limpieza finalizada para todas las fuentes.")
        return cleaned_dataframes

    def _clean_clientes(
        self,
        clientes_df: pd.DataFrame,
        estimador_ing_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Limpia tabla clientes con reglas documentadas en el EDA."""
        df = clientes_df.copy()

        rows_before = len(df)
        duplicated_before = int(df.duplicated(subset=["numero_id"], keep=False).sum())
        df = df.drop_duplicates(subset=["numero_id"], keep="first")
        rows_after = len(df)
        logging.info(
            "clientes -> duplicados por numero_id detectados: %s | filas eliminadas: %s",
            f"{duplicated_before:,}",
            f"{(rows_before - rows_after):,}",
        )

        # Bandera antes de imputar ingresos para preservar semantica de falta de informacion.
        df["flag_sin_info_financiera"] = df["ingresos_mensuales"].isna().astype(int)

        # Imputacion de ingresos_mensuales en dos pasos: estimador_ing y luego 0.
        est_agg = (
            estimador_ing_df.groupby("numero_id", as_index=False)["estimador_ingreso"]
            .mean()
            .rename(columns={"estimador_ingreso": "estimador_ingreso_aux"})
        )
        df = df.merge(est_agg, on="numero_id", how="left")

        ingresos_nulos_before = int(df["ingresos_mensuales"].isna().sum())
        df["ingresos_mensuales"] = df["ingresos_mensuales"].fillna(df["estimador_ingreso_aux"])
        ingresos_nulos_after_est = int(df["ingresos_mensuales"].isna().sum())

        rescued_with_estimador = ingresos_nulos_before - ingresos_nulos_after_est
        df["ingresos_mensuales"] = df["ingresos_mensuales"].fillna(0)
        filled_with_zero = ingresos_nulos_after_est
        df = df.drop(columns=["estimador_ingreso_aux"])

        logging.info(
            "clientes -> ingresos nulos iniciales: %s | imputados desde estimador_ing: %s | imputados en 0: %s",
            f"{ingresos_nulos_before:,}",
            f"{rescued_with_estimador:,}",
            f"{filled_with_zero:,}",
        )

        # Imputacion a 0 para bloque financiero y recalculo patrimonial.
        for column in ["total_egresos_mensuales", "total_activos", "total_pasivos"]:
            nulls_before = int(df[column].isna().sum())
            df[column] = df[column].fillna(0)
            logging.info(
                "clientes -> columna '%s' imputada a 0 en %s filas.",
                column,
                f"{nulls_before:,}",
            )

        df["total_patrimonio"] = df["total_activos"] - df["total_pasivos"]
        logging.info("clientes -> total_patrimonio recalculado como total_activos - total_pasivos.")

        # Reglas categoricas del EDA.
        if "desc_tipo_de_vivienda" in df.columns:
            df = df.drop(columns=["desc_tipo_de_vivienda"])
            logging.info("clientes -> columna 'desc_tipo_de_vivienda' eliminada por alta ausencia de datos.")

        genero_nulls = int(df["desc_genero"].isna().sum())
        df["desc_genero"] = df["desc_genero"].fillna("NO_REGISTRADO")
        logging.info(
            "clientes -> desc_genero imputado con 'NO_REGISTRADO' en %s filas.",
            f"{genero_nulls:,}",
        )

        total_nulls = int(df.isna().sum().sum())
        logging.info("clientes -> nulos totales luego de limpieza: %s", f"{total_nulls:,}")

        return df

    def _clean_crean_aho_cte(self, creates_aho_cte_df: pd.DataFrame) -> pd.DataFrame:
        """Descarta registros con saldos negativos en cuentas de ahorro/corriente."""
        df = creates_aho_cte_df.copy()

        productos_bancarios = {"CUENTA DE AHORRO", "CUENTA DE CORRIENTE"}
        mask_producto = df["producto"].astype(str).str.upper().isin(productos_bancarios)
        mask_negativo = df["saldo"] < 0
        rows_to_drop = int((mask_producto & mask_negativo).sum())

        if rows_to_drop > 0:
            df = df.loc[~(mask_producto & mask_negativo)].copy()

        logging.info(
            "crean_aho_cte -> registros descartados por saldo negativo (ahorro/corriente): %s",
            f"{rows_to_drop:,}",
        )

        return df


def _print_null_report(cleaned_dataframes: Dict[str, pd.DataFrame]) -> None:
    """Imprime reporte de validacion de nulos por fuente y total general."""
    print("\n=== REPORTE DE VERIFICACION DE NULOS ===")
    global_nulls = 0

    for source_name, dataframe in cleaned_dataframes.items():
        nulls = int(dataframe.isna().sum().sum())
        global_nulls += nulls
        status = "OK" if nulls == 0 else "PENDIENTE"
        print(
            f"[{status}] {source_name:<22} "
            f"filas={len(dataframe):>10,} | columnas={dataframe.shape[1]:>2} | nulos={nulls:,}"
        )

    print("-" * 88)
    print(f"TOTAL NULOS EN TODAS LAS FUENTES: {global_nulls:,}")


if __name__ == "__main__":
    from extraction import DataExtractor

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_PATH = BASE_DIR / "data"

    logging.info("Buscando datos fuente en: %s", RAW_DATA_PATH)

    extractor = DataExtractor(data_dir=RAW_DATA_PATH)
    raw_dfs = extractor.extract_all_sources()

    cleaner = DataCleaner(raw_dataframes=raw_dfs)
    clean_dfs = cleaner.clean_all_sources()

    _print_null_report(cleaned_dataframes=clean_dfs)
