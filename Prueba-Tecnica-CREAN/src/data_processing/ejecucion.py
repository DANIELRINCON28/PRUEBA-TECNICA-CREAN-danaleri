"""
src/data_processing/ejecucion.py
Orquestador del pipeline por steps para ejecutar el flujo completo desde un solo script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import pandas as pd

from extraction import DataExtractor
from cleaning import DataCleaner


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


class PipelineExecutor:
	"""Ejecuta el pipeline de datos en pasos secuenciales y trazables."""

	def __init__(self, data_dir: Path):
		self.data_dir = Path(data_dir)
		self.raw_dataframes: Dict[str, pd.DataFrame] | None = None
		self.cleaned_dataframes: Dict[str, pd.DataFrame] | None = None
		self._steps: Dict[str, Callable[..., Any]] = {
			"extract_data": self.step_extract_data,
			"clean_data": self.step_clean_data,
			"print_summary": self.step_print_summary,
		}

	def step_extract_data(self) -> Dict[str, pd.DataFrame]:
		"""Step: extrae todas las fuentes SQLite a DataFrames en memoria."""
		logging.info("[STEP] extract_data -> iniciado")
		extractor = DataExtractor(data_dir=self.data_dir)
		self.raw_dataframes = extractor.extract_all_sources()
		logging.info(
			"[STEP] extract_data -> completado. Fuentes cargadas: %s",
			", ".join(sorted(self.raw_dataframes.keys())),
		)
		return self.raw_dataframes

	def step_clean_data(self) -> Dict[str, pd.DataFrame]:
		"""Step: aplica reglas de limpieza de negocio sobre las fuentes extraidas."""
		if self.raw_dataframes is None:
			raise RuntimeError("No se puede limpiar sin extraccion previa. Ejecuta extract_data primero.")

		logging.info("[STEP] clean_data -> iniciado")
		cleaner = DataCleaner(raw_dataframes=self.raw_dataframes)
		self.cleaned_dataframes = cleaner.clean_all_sources()
		logging.info("[STEP] clean_data -> completado")
		return self.cleaned_dataframes

	def step_print_summary(self) -> None:
		"""Step: imprime resumen de salida para validacion rapida."""
		if self.cleaned_dataframes is None:
			raise RuntimeError("No hay datos limpios para resumir. Ejecuta clean_data primero.")
		self._print_summary(self.cleaned_dataframes)

	def run_steps(self, steps: Iterable[dict[str, Any]]) -> Dict[str, Any]:
		"""
		Ejecuta una lista de steps dinamicos.

		Cada step debe tener esta forma:
		{"name": "extract_data", "kwargs": {}}
		"""
		results: Dict[str, Any] = {}

		for idx, step in enumerate(steps, start=1):
			name = step["name"]
			kwargs = step.get("kwargs", {})

			if name not in self._steps:
				raise KeyError(f"Step no soportado: {name}. Steps validos: {sorted(self._steps.keys())}")

			logging.info("[RUN] Step %s -> %s", idx, name)
			result = self._steps[name](**kwargs)
			results[name] = result

		return results

	def run(self) -> Dict[str, pd.DataFrame]:
		"""Ejecuta el flujo default del pipeline usando steps configurables por kwargs."""
		logging.info("=== INICIO ORQUESTACION PIPELINE ===")
		default_steps = [
			{"name": "extract_data", "kwargs": {}},
			{"name": "clean_data", "kwargs": {}},
			{"name": "print_summary", "kwargs": {}},
		]
		self.run_steps(default_steps)
		logging.info("=== FIN ORQUESTACION PIPELINE ===")

		if self.cleaned_dataframes is None:
			raise RuntimeError("El pipeline finalizo sin generar datos limpios.")
		return self.cleaned_dataframes

	@staticmethod
	def _print_summary(cleaned_dataframes: Dict[str, pd.DataFrame]) -> None:
		"""Imprime resumen de salida del pipeline para validacion rapida."""
		print("\n=== RESUMEN FINAL DEL PIPELINE ===")
		total_rows = 0
		total_nulls = 0

		for source_name, dataframe in cleaned_dataframes.items():
			rows = len(dataframe)
			nulls = int(dataframe.isna().sum().sum())
			total_rows += rows
			total_nulls += nulls
			print(
				f"- {source_name:<22} filas={rows:>10,} | columnas={dataframe.shape[1]:>2} | nulos={nulls:,}"
			)

		print("-" * 88)
		print(f"TOTAL FILAS (sumatoria fuentes): {total_rows:,}")
		print(f"TOTAL NULOS (sumatoria fuentes): {total_nulls:,}")


if __name__ == "__main__":
	BASE_DIR = Path(__file__).resolve().parent.parent.parent
	RAW_DATA_PATH = BASE_DIR / "data"

	executor = PipelineExecutor(data_dir=RAW_DATA_PATH)
	executor.run()