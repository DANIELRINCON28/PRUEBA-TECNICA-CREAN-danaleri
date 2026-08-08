"""
src/data_processing/feature_engineering.py
Modulo de creacion de caracteristicas y consolidacion del Master Feature Store.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from cleaning import DataCleaner
from extraction import DataExtractor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class FeatureEngineer:
    """Construye y consolida el Master Feature Store centrado en clientes."""

    REQUIRED_SOURCES = {
        "clientes",
        "crean_aho_cte",
        "crean_bolsillos",
        "crean_fiducuenta",
        "crean_inv_virtual_cdt",
        "estimador_ing",
        "invesbot",
    }

    EXPECTED_MASTER_ROWS = 860_223

    def __init__(self, cleaned_dataframes: Dict[str, pd.DataFrame]):
        """Inicializa el generador de features con la salida de DataCleaner."""
        missing_sources = self.REQUIRED_SOURCES.difference(cleaned_dataframes.keys())
        if missing_sources:
            raise KeyError(
                "Faltan fuentes requeridas para feature engineering: "
                f"{sorted(missing_sources)}"
            )
        self.cleaned_dataframes = cleaned_dataframes

    def build_master_feature_store(self) -> pd.DataFrame:
        """Construye el DataFrame maestro con features y targets por cliente."""
        logging.info("Iniciando construccion de Master Feature Store...")

        df_master = self._build_client_base_features(self.cleaned_dataframes["clientes"])

        agg_aho_cte = self._aggregate_crean_aho_cte(self.cleaned_dataframes["crean_aho_cte"])
        agg_bolsillos = self._aggregate_crean_bolsillos(self.cleaned_dataframes["crean_bolsillos"])
        agg_fiducuenta = self._aggregate_crean_fiducuenta(self.cleaned_dataframes["crean_fiducuenta"])
        agg_invesbot = self._aggregate_invesbot(self.cleaned_dataframes["invesbot"])
        agg_cdt = self._aggregate_crean_inv_virtual_cdt(
            self.cleaned_dataframes["crean_inv_virtual_cdt"]
        )

        # LEFT JOIN secuencial, preservando la entidad maestra de clientes (1 fila por numero_id).
        for agg_df in [agg_aho_cte, agg_bolsillos, agg_fiducuenta, agg_invesbot, agg_cdt]:
            df_master = df_master.merge(agg_df, on="numero_id", how="left", validate="one_to_one")

        continuous_product_features = [
            "saldo_prom_liquidez",
            "saldo_max_liquidez",
            "cant_cuentas_aho",
            "saldo_total_bolsillos",
            "cant_bolsillos",
            "saldo_prom_fiducuenta",
            "saldo_prom_invesbot",
            "saldo_prom_inv_cdt",
            "saldo_max_inv_cdt",
        ]
        flag_features = [
            "flag_tiene_aho",
            "flag_tiene_bolsillos",
            "flag_tiene_fiducuenta",
            "flag_tiene_invesbot",
            "flag_tiene_cdt",
        ]

        df_master[continuous_product_features] = df_master[continuous_product_features].fillna(0.0)
        df_master[flag_features] = df_master[flag_features].fillna(0).astype("int64")

        # Feature set avanzado para valor y propension.
        df_master["num_productos_activos"] = (
            df_master["flag_tiene_aho"]
            + df_master["flag_tiene_bolsillos"]
            + df_master["flag_tiene_fiducuenta"]
            + df_master["flag_tiene_invesbot"]
            + df_master["flag_tiene_cdt"]
        )
        df_master["saldo_total_inversiones"] = (
            df_master["saldo_prom_fiducuenta"]
            + df_master["saldo_prom_invesbot"]
            + df_master["saldo_prom_inv_cdt"]
        )
        df_master["ratio_liquidez_vs_ingreso"] = (
            df_master["saldo_prom_liquidez"] / (df_master["ingresos_mensuales"] + 1)
        )
        df_master["penetracion_bolsillos_vs_liquidez"] = (
            df_master["saldo_total_bolsillos"] / (df_master["saldo_prom_liquidez"] + 1)
        )
        df_master["flag_superavit_operativo"] = (
            df_master["ingresos_mensuales"] > (df_master["total_egresos_mensuales"] * 1.5)
        ).astype("int64")
        df_master["flag_propension_digital_previa"] = (
            (df_master["flag_tiene_invesbot"] == 1) | (df_master["saldo_prom_inv_cdt"] > 0)
        ).astype("int64")

        # Targets de negocio para clasificacion (adopcion) y regresion (monto potencial 12M).
        df_master["target_adopcion"] = (
            (df_master["saldo_total_inversiones"] > 0) | (df_master["flag_tiene_invesbot"] == 1)
        ).astype("int64")
        excedente_liquidez = np.maximum(
            0,
            df_master["saldo_prom_liquidez"] - (df_master["total_egresos_mensuales"] * 2),
        )
        df_master["target_monto_12m"] = df_master["saldo_total_inversiones"] + excedente_liquidez

        self._validate_master(df_master)

        logging.info(
            "Master Feature Store construido correctamente: %s filas | %s columnas | nulos totales=%s",
            f"{len(df_master):,}",
            f"{df_master.shape[1]:,}",
            "0",
        )

        return df_master

    @staticmethod
    def _build_client_base_features(clientes_df: pd.DataFrame) -> pd.DataFrame:
        """Calcula ratios financieros base sobre la entidad maestra clientes."""
        df = clientes_df.copy()

        df["margen_libre_estimado"] = (
            df["ingresos_mensuales"] - df["total_egresos_mensuales"]
        )
        df["ratio_apalancamiento"] = df["total_pasivos"] / (df["total_activos"] + 1)
        df["ratio_cobertura_egresos"] = (
            df["total_activos"] / (df["total_egresos_mensuales"] * 12 + 1)
        )

        return df

    @staticmethod
    def _aggregate_crean_aho_cte(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega metricas de liquidez y bandera de posesion desde cuentas de ahorro/corriente."""
        agg_df = (
            df.groupby("numero_id", as_index=False)
            .agg(
                saldo_prom_liquidez=("saldo", "mean"),
                saldo_max_liquidez=("saldo", "max"),
                cant_cuentas_aho=("producto", "nunique"),
            )
            .copy()
        )
        agg_df["flag_tiene_aho"] = 1
        return agg_df

    @staticmethod
    def _aggregate_crean_bolsillos(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega metricas de bolsillos y bandera de posesion."""
        agg_df = (
            df.groupby("numero_id", as_index=False)
            .agg(
                saldo_total_bolsillos=("saldo", "mean"),
                cant_bolsillos=("saldo", "count"),
            )
            .copy()
        )
        agg_df["flag_tiene_bolsillos"] = 1
        return agg_df

    @staticmethod
    def _aggregate_crean_fiducuenta(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega saldo promedio de fiducuenta y su bandera de posesion."""
        agg_df = (
            df.groupby("numero_id", as_index=False)
            .agg(saldo_prom_fiducuenta=("saldo", "mean"))
            .copy()
        )
        agg_df["flag_tiene_fiducuenta"] = 1
        return agg_df

    @staticmethod
    def _aggregate_invesbot(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega saldo promedio de invesbot y su bandera de posesion."""
        agg_df = (
            df.groupby("numero_id", as_index=False)
            .agg(saldo_prom_invesbot=("saldo", "mean"))
            .copy()
        )
        agg_df["flag_tiene_invesbot"] = 1
        return agg_df

    @staticmethod
    def _aggregate_crean_inv_virtual_cdt(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega saldo promedio/maximo de inversion CDT y su bandera de posesion."""
        agg_df = (
            df.groupby("numero_id", as_index=False)
            .agg(
                saldo_prom_inv_cdt=("saldo", "mean"),
                saldo_max_inv_cdt=("saldo", "max"),
            )
            .copy()
        )
        agg_df["flag_tiene_cdt"] = 1
        return agg_df

    @classmethod
    def _validate_master(cls, df_master: pd.DataFrame) -> None:
        """Valida integridad minima del Feature Store: cardinalidad y nulos."""
        if len(df_master) != cls.EXPECTED_MASTER_ROWS:
            raise ValueError(
                "El Master Feature Store no conserva la cardinalidad esperada de clientes. "
                f"Esperado={cls.EXPECTED_MASTER_ROWS:,} | Actual={len(df_master):,}"
            )

        total_nulls = int(df_master.isna().sum().sum())
        if total_nulls != 0:
            raise ValueError(
                "El Master Feature Store contiene nulos despues de la consolidacion. "
                f"Nulos detectados={total_nulls:,}"
            )


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_PATH = BASE_DIR / "data"
    FEATURE_STORE_DIR = RAW_DATA_PATH / "feature_store"
    FEATURE_STORE_PATH = FEATURE_STORE_DIR / "df_master.parquet"

    logging.info("Buscando datos fuente en: %s", RAW_DATA_PATH)

    extractor = DataExtractor(data_dir=RAW_DATA_PATH)
    raw_dfs = extractor.extract_all_sources()

    cleaner = DataCleaner(raw_dataframes=raw_dfs)
    cleaned_dfs = cleaner.clean_all_sources()

    engineer = FeatureEngineer(cleaned_dataframes=cleaned_dfs)
    df_master = engineer.build_master_feature_store()

    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    df_master.to_parquet(FEATURE_STORE_PATH, index=False)

    logging.info(
        "Feature Store guardado en %s | filas=%s | columnas=%s",
        FEATURE_STORE_PATH,
        f"{len(df_master):,}",
        f"{df_master.shape[1]:,}",
    )
