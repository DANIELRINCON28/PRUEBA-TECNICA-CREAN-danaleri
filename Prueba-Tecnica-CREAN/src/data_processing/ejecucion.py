"""
src/data_processing/ejecucion.py
Orquestador del pipeline por steps para ejecutar el flujo completo desde un solo script.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import webbrowser
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable

import pandas as pd

from extraction import DataExtractor
from cleaning import DataCleaner
from feature_engineering import FeatureEngineer


BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from models.train import TrainingArtifacts, train_pipeline


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


class PipelineExecutor:
	"""Ejecuta el pipeline de datos en pasos secuenciales y trazables."""

	LINE_WIDTH = 104
	STEP_TITLES = {
		"extract_data": "Extraccion de fuentes",
		"clean_data": "Limpieza de datos",
		"feature_engineering": "Creacion de features",
		"train_models": "Entrenamiento de modelos",
		"launch_dashboard": "Despliegue de dashboard Streamlit",
		"print_summary": "Resumen de resultados",
	}

	def __init__(self, data_dir: Path):
		self.data_dir = Path(data_dir)
		self.raw_dataframes: Dict[str, pd.DataFrame] | None = None
		self.cleaned_dataframes: Dict[str, pd.DataFrame] | None = None
		self.master_feature_store: pd.DataFrame | None = None
		self.training_artifacts: TrainingArtifacts | None = None
		self._steps: Dict[str, Callable[..., Any]] = {
			"extract_data": self.step_extract_data,
			"clean_data": self.step_clean_data,
			"feature_engineering": self.step_feature_engineering,
			"train_models": self.step_train_models,
			"launch_dashboard": self.step_launch_dashboard,
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

	def step_feature_engineering(self, save_output: bool = True) -> pd.DataFrame:
		"""Step: crea el Master Feature Store a partir de fuentes limpias."""
		if self.cleaned_dataframes is None:
			raise RuntimeError("No se puede crear features sin limpieza previa. Ejecuta clean_data primero.")

		logging.info("[STEP] feature_engineering -> iniciado")
		engineer = FeatureEngineer(cleaned_dataframes=self.cleaned_dataframes)
		self.master_feature_store = engineer.build_master_feature_store()

		if save_output:
			feature_store_dir = self.data_dir / "feature_store"
			feature_store_path = feature_store_dir / "df_master.parquet"
			feature_store_dir.mkdir(parents=True, exist_ok=True)
			self.master_feature_store.to_parquet(feature_store_path, index=False)
			logging.info(
				"[STEP] feature_engineering -> parquet guardado en %s",
				feature_store_path,
			)

		logging.info(
			"[STEP] feature_engineering -> completado. Master: %s filas | %s columnas",
			f"{len(self.master_feature_store):,}",
			f"{self.master_feature_store.shape[1]:,}",
		)
		return self.master_feature_store

	def step_print_summary(self) -> None:
		"""Step: imprime resumen de salida para validacion rapida."""
		if self.cleaned_dataframes is None:
			raise RuntimeError("No hay datos limpios para resumir. Ejecuta clean_data primero.")
		self._print_summary(self.cleaned_dataframes)

		if self.master_feature_store is not None:
			self._print_master_summary(self.master_feature_store)

		if self.training_artifacts is not None:
			self._print_training_summary(self.training_artifacts)

	def step_train_models(self) -> TrainingArtifacts:
		"""Step: entrena modelos de adopcion, monto y scoring de negocio."""
		if self.master_feature_store is None:
			raise RuntimeError(
				"No se puede entrenar sin feature store. Ejecuta feature_engineering primero."
			)

		logging.info("[STEP] train_models -> iniciado")
		self.training_artifacts = train_pipeline(self.master_feature_store)
		logging.info("[STEP] train_models -> completado")
		logging.info(
			"Artefactos listos para consumo: models/*.pkl y data/scores/df_predictions.parquet"
		)
		return self.training_artifacts

	def step_launch_dashboard(self, detach: bool = True) -> str:
		"""Step: lanza la interfaz Streamlit para consumo del negocio."""
		dashboard_path = SRC_DIR / "dashboard" / "app.py"
		scores_path = self.data_dir / "scores" / "df_predictions.parquet"
		dashboard_url = "http://localhost:8501"

		if not dashboard_path.exists():
			raise FileNotFoundError(
				f"No se encontro el dashboard en {dashboard_path}."
			)

		if not scores_path.exists():
			raise FileNotFoundError(
				"No se encontro data/scores/df_predictions.parquet. "
				"Ejecuta train_models antes de lanzar el dashboard."
			)

		command = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
		logging.info("[STEP] launch_dashboard -> iniciado")
		logging.info("Comando dashboard: %s", " ".join(command))

		if detach:
			process = subprocess.Popen(command, cwd=str(BASE_DIR))
			logging.info("[STEP] launch_dashboard -> proceso iniciado en background")
			logging.info("Dashboard disponible en: %s", dashboard_url)
			try:
				webbrowser.open(dashboard_url)
			except Exception as exc:
				logging.warning("No fue posible abrir el navegador automaticamente: %s", exc)
			return f"Dashboard lanzado en background (PID={process.pid})"

		subprocess.run(command, cwd=str(BASE_DIR), check=True)
		logging.info("[STEP] launch_dashboard -> proceso finalizado")
		return "Dashboard ejecutado en foreground"

	def run_steps(self, steps: Iterable[dict[str, Any]]) -> Dict[str, Any]:
		"""
		Ejecuta una lista de steps dinamicos.

		Cada step debe tener esta forma:
		{"name": "extract_data", "kwargs": {}}
		"""
		results: Dict[str, Any] = {}
		steps_list = list(steps)

		self._print_section_header("PLAN DE EJECUCION")
		for idx, step in enumerate(steps_list, start=1):
			name = step["name"]
			title = self.STEP_TITLES.get(name, name)
			print(f"{idx:>2}. {name:<20} | {title}")
		print("-" * self.LINE_WIDTH)

		pipeline_start = perf_counter()

		for idx, step in enumerate(steps_list, start=1):
			name = step["name"]
			kwargs = step.get("kwargs", {})

			if name not in self._steps:
				raise KeyError(f"Step no soportado: {name}. Steps validos: {sorted(self._steps.keys())}")

			title = self.STEP_TITLES.get(name, name)
			self._print_step_start(idx=idx, total=len(steps_list), name=name, title=title)
			step_start = perf_counter()
			result = self._steps[name](**kwargs)
			step_elapsed = perf_counter() - step_start
			self._print_step_end(name=name, elapsed_seconds=step_elapsed, result=result)
			results[name] = result

		total_elapsed = perf_counter() - pipeline_start
		self._print_section_header("EJECUCION COMPLETADA")
		print(f"Duracion total: {total_elapsed:,.2f} s")
		print("-" * self.LINE_WIDTH)

		return results

	def run(self, launch_dashboard: bool = False) -> pd.DataFrame:
		"""Ejecuta el flujo default del pipeline usando steps configurables por kwargs."""
		self._print_section_header("INICIO ORQUESTADOR PIPELINE")
		print(f"Directorio de datos: {self.data_dir}")
		print("-" * self.LINE_WIDTH)

		logging.info("Inicio de orquestacion de pipeline")
		default_steps = [
			{"name": "extract_data", "kwargs": {}},
			{"name": "clean_data", "kwargs": {}},
			{"name": "feature_engineering", "kwargs": {"save_output": True}},
			{"name": "train_models", "kwargs": {}},
			{"name": "print_summary", "kwargs": {}},
		]
		if launch_dashboard:
			default_steps.append({"name": "launch_dashboard", "kwargs": {"detach": True}})
			logging.info("Dashboard habilitado: se lanzara Streamlit al finalizar el pipeline")
		self.run_steps(default_steps)
		logging.info("Fin de orquestacion de pipeline")

		if self.master_feature_store is None:
			raise RuntimeError("El pipeline finalizo sin generar el Master Feature Store.")
		return self.master_feature_store

	@staticmethod
	def _print_summary(cleaned_dataframes: Dict[str, pd.DataFrame]) -> None:
		"""Imprime resumen de salida del pipeline para validacion rapida."""
		print("\n" + "=" * PipelineExecutor.LINE_WIDTH)
		print("RESUMEN FINAL DE FUENTES LIMPIAS")
		print("=" * PipelineExecutor.LINE_WIDTH)
		print(f"{'Fuente':<24} {'Filas':>14} {'Columnas':>10} {'Nulos':>14}")
		print("-" * PipelineExecutor.LINE_WIDTH)
		total_rows = 0
		total_nulls = 0

		for source_name, dataframe in cleaned_dataframes.items():
			rows = len(dataframe)
			nulls = int(dataframe.isna().sum().sum())
			total_rows += rows
			total_nulls += nulls
			print(
				f"{source_name:<24} {rows:>14,} {dataframe.shape[1]:>10,} {nulls:>14,}"
			)

		print("-" * PipelineExecutor.LINE_WIDTH)
		print(f"{'TOTAL':<24} {total_rows:>14,} {'-':>10} {total_nulls:>14,}")

	@staticmethod
	def _print_master_summary(df_master: pd.DataFrame) -> None:
		"""Imprime resumen del Master Feature Store consolidado."""
		nulls = int(df_master.isna().sum().sum())
		print("\n" + "=" * PipelineExecutor.LINE_WIDTH)
		print("MASTER FEATURE STORE")
		print("=" * PipelineExecutor.LINE_WIDTH)
		print(f"{'Filas':<30}: {len(df_master):,}")
		print(f"{'Columnas':<30}: {df_master.shape[1]:,}")
		print(f"{'Nulos totales':<30}: {nulls:,}")
		print("-" * PipelineExecutor.LINE_WIDTH)

	@staticmethod
	def _print_section_header(title: str) -> None:
		"""Imprime un encabezado de seccion estandar para mejorar legibilidad."""
		print("\n" + "=" * PipelineExecutor.LINE_WIDTH)
		print(title)
		print("=" * PipelineExecutor.LINE_WIDTH)

	@staticmethod
	def _print_step_start(idx: int, total: int, name: str, title: str) -> None:
		"""Imprime inicio de step con indice, nombre tecnico y descripcion."""
		print("\n" + "-" * PipelineExecutor.LINE_WIDTH)
		print(f"STEP {idx}/{total} | {name}")
		print(f"Descripcion: {title}")
		print("-" * PipelineExecutor.LINE_WIDTH)

	@staticmethod
	def _print_step_end(name: str, elapsed_seconds: float, result: Any) -> None:
		"""Imprime cierre de step con duracion y metrica principal del resultado."""
		metric = PipelineExecutor._format_step_result_metric(name=name, result=result)
		print(f"Estado: OK | Duracion: {elapsed_seconds:,.2f} s")
		if metric:
			print(f"Salida : {metric}")

	@staticmethod
	def _format_step_result_metric(name: str, result: Any) -> str:
		"""Resume el resultado de un step para mostrarlo en consola."""
		if isinstance(result, pd.DataFrame):
			return f"DataFrame {len(result):,} filas x {result.shape[1]:,} columnas"

		if isinstance(result, TrainingArtifacts):
			classifier = result.classifier_eval.test_metrics
			regressor = result.regressor_eval.test_metrics
			return (
				"Clasificacion ROC-AUC="
				f"{classifier['roc_auc']:.4f} | "
				"Regresion R2="
				f"{regressor['r2']:.4f} | "
				f"Predicciones={len(result.predictions):,}"
			)

		if isinstance(result, dict):
			if all(isinstance(value, pd.DataFrame) for value in result.values()):
				total_rows = sum(len(value) for value in result.values())
				return (
					f"{len(result):,} fuentes ({total_rows:,} filas agregadas en memoria)"
				)
			return f"Diccionario con {len(result):,} elementos"

		if result is None:
			return "Sin objeto de retorno"

		return f"Tipo de salida: {type(result).__name__}"

	@staticmethod
	def _print_training_summary(artifacts: TrainingArtifacts) -> None:
		"""Imprime un resumen ejecutivo del entrenamiento y del scoring generado."""
		classifier_train = artifacts.classifier_eval.train_metrics
		classifier_test = artifacts.classifier_eval.test_metrics
		regressor_train = artifacts.regressor_eval.train_metrics
		regressor_test = artifacts.regressor_eval.test_metrics
		decile_distribution = (
			artifacts.predictions.groupby("decel_prioridad", as_index=False)
			.agg(
				clientes=("numero_id", "count"),
				valor_esperado_total=("valor_esperado_12m", "sum"),
			)
			.sort_values("decel_prioridad", ascending=False)
		)

		print("\n" + "=" * PipelineExecutor.LINE_WIDTH)
		print("RESUMEN DE ENTRENAMIENTO Y SCORING")
		print("=" * PipelineExecutor.LINE_WIDTH)
		print(
			"Clasificacion train/test | "
			f"ROC-AUC {classifier_train['roc_auc']:.4f}/{classifier_test['roc_auc']:.4f} | "
			f"PR-AUC {classifier_train['pr_auc']:.4f}/{classifier_test['pr_auc']:.4f} | "
			f"F1 {classifier_train['f1']:.4f}/{classifier_test['f1']:.4f}"
		)
		print(
			"Regresion train/test     | "
			f"MAE {regressor_train['mae']:.2f}/{regressor_test['mae']:.2f} | "
			f"RMSE {regressor_train['rmse']:.2f}/{regressor_test['rmse']:.2f} | "
			f"R2 {regressor_train['r2']:.4f}/{regressor_test['r2']:.4f}"
		)
		print(f"{'Predicciones generadas':<30}: {len(artifacts.predictions):,}")
		print(f"{'Artefactos modelos':<30}: models/lgbm_adopcion.pkl | models/lgbm_monto.pkl")
		print(f"{'Archivo scoring':<30}: data/scores/df_predictions.parquet")
		print("-" * PipelineExecutor.LINE_WIDTH)
		print(f"{'Decil':<8} {'Clientes':>14} {'EV total':>22}")
		print("-" * PipelineExecutor.LINE_WIDTH)
		for _, row in decile_distribution.iterrows():
			print(
				f"{int(row['decel_prioridad']):<8} "
				f"{int(row['clientes']):>14,} "
				f"{float(row['valor_esperado_total']):>22,.2f}"
			)
		print("-" * PipelineExecutor.LINE_WIDTH)


if __name__ == "__main__":
	BASE_DIR = Path(__file__).resolve().parent.parent.parent
	RAW_DATA_PATH = BASE_DIR / "data"

	executor = PipelineExecutor(data_dir=RAW_DATA_PATH)
	executor.run(launch_dashboard=True)