"""
src/data_processing/extract.py
Módulo para la extracción modular y segura de bases de datos SQLite (.db) a DataFrames de Pandas.
"""

from pathlib import Path
import logging
import sqlite3
import pandas as pd

# Configuración de logs para trazabilidad en producción
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataExtractor:
    """Clase encargada de conectar y extraer tablas desde archivos SQLite .db."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"El directorio especificado no existe: {self.data_dir.resolve()}")

        # Mapeo de nombres lógicos a nombres de archivo .db
        self.sources = {
            "clientes": "clientes.db",
            "crean_aho_cte": "crean_aho_cte.db",
            "crean_bolsillos": "crean_bolsillos.db",
            "crean_fiducuenta": "crean_fiducuenta.db",
            "crean_inv_virtual_cdt": "crean_inv_virtual_cdt.db",
            "estimador_ing": "estimador_ing.db",
            "invesbot": "invesbot.db",
        }

    def _get_table_name(self, conn: sqlite3.Connection) -> str:
        """Obtiene dinámicamente el nombre de la tabla principal dentro del archivo .db."""
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, conn)
        if tables.empty:
            raise ValueError("No se encontraron tablas dentro de la base de datos.")
        return tables.iloc[0, 0]

    def extract_single_table(self, source_key: str) -> pd.DataFrame:
        """Extrae una fuente específica por su clave identificadora."""
        if source_key not in self.sources:
            raise KeyError(f"La clave '{source_key}' no está definida en las fuentes.")

        db_path = self.data_dir / self.sources[source_key]
        if not db_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo SQLite: {db_path.resolve()}")

        logging.info(f"Extrayendo tabla desde: {self.sources[source_key]}...")
        with sqlite3.connect(db_path) as conn:
            table_name = self._get_table_name(conn)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        logging.info(f" -> '{source_key}' cargada exitosamente: {len(df):,} filas, {len(df.columns)} columnas.")
        return df

    def extract_all_sources(self) -> dict[str, pd.DataFrame]:
        """Extrae todas las 7 fuentes de datos y las retorna en un diccionario de DataFrames."""
        logging.info("Iniciando extracción masiva de las 7 fuentes de datos...")
        dataframes = {}
        for key in self.sources.keys():
            dataframes[key] = self.extract_single_table(key)
        
        logging.info("Extracción de todas las fuentes finalizada correctamente.")
        return dataframes


if __name__ == "__main__":
    # Resuelve la ruta 'data' ubicándola 2 niveles arriba respecto a la ubicación de este script (src/data_processing/)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_PATH = BASE_DIR / "data"
    
    print(f"Buscando carpeta de datos en: {RAW_DATA_PATH}")
    
    extractor = DataExtractor(data_dir=RAW_DATA_PATH)
    raw_dfs = extractor.extract_all_sources()