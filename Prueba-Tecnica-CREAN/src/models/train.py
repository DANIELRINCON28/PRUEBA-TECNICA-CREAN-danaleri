"""
src/models/train.py
Entrenamiento de modelos para adopcion y monto potencial de inversion.

El pipeline implementa una arquitectura two-part (hurdle):
- Clasificacion binaria para propension de adopcion.
- Regresion condicionada para monto potencial a 12 meses.
- Scoring de valor esperado para priorizacion comercial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_ADOPTION = "target_adopcion"
TARGET_AMOUNT = "target_monto_12m"
ID_COLUMN = "numero_id"

AMOUNT_FEATURE_COLUMNS: List[str] = [
    "grupo_edad",
    "desc_genero",
    "desc_segmento",
    "ingresos_mensuales",
    "total_egresos_mensuales",
    "total_activos",
    "total_pasivos",
    "total_patrimonio",
    "flag_sin_info_financiera",
    "margen_libre_estimado",
    "ratio_apalancamiento",
    "ratio_cobertura_egresos",
    "ratio_liquidez_vs_ingreso",
    "penetracion_bolsillos_vs_liquidez",
    "flag_superavit_operativo",
    "saldo_prom_liquidez",
    "saldo_max_liquidez",
    "cant_cuentas_aho",
    "saldo_total_bolsillos",
    "cant_bolsillos",
    "flag_tiene_aho",
    "flag_tiene_bolsillos",
    "flag_tiene_fiducuenta",
    "flag_tiene_invesbot",
    "flag_tiene_cdt",
    "num_productos_activos",
    "flag_propension_digital_previa",
]

ADOPTION_FEATURE_COLUMNS: List[str] = [
    column
    for column in AMOUNT_FEATURE_COLUMNS
    if column
    not in {
        "flag_tiene_fiducuenta",
        "flag_tiene_invesbot",
        "flag_tiene_cdt",
        "num_productos_activos",
        "flag_propension_digital_previa",
    }
]

LEAKAGE_COLUMNS = {
    ID_COLUMN,
    TARGET_ADOPTION,
    TARGET_AMOUNT,
    "saldo_total_inversiones",
    "flag_adopcion_invesbot",
    "excedente_liquidez",
}

CAT_FEATURES = ["grupo_edad", "desc_genero", "desc_segmento"]


@dataclass
class EvaluationBundle:
    """Contiene el modelo usado para evaluacion y sus metricas."""

    model: Pipeline
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


@dataclass
class TrainingArtifacts:
    """Agrupa modelos finales, metricas y scoring generado."""

    classifier_eval: EvaluationBundle
    regressor_eval: EvaluationBundle
    final_classifier: Pipeline
    final_regressor: Pipeline
    predictions: pd.DataFrame


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _feature_store_path() -> Path:
    return _project_root() / "data" / "feature_store" / "df_master.parquet"


def _models_dir() -> Path:
    return _project_root() / "models"


def _scores_path() -> Path:
    return _project_root() / "data" / "scores" / "df_predictions.parquet"


def _build_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    numeric_features = [column for column in feature_columns if column not in CAT_FEATURES]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CAT_FEATURES),
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def _validate_schema(df: pd.DataFrame) -> None:
    required_columns = set(AMOUNT_FEATURE_COLUMNS) | {ID_COLUMN, TARGET_ADOPTION, TARGET_AMOUNT}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(
            "El feature store no contiene las columnas requeridas para entrenamiento: "
            f"{sorted(missing_columns)}"
        )


def _build_xy(df: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    _validate_schema(df)

    X = df[list(feature_columns)].copy()
    y_adoption = df[TARGET_ADOPTION].astype(int).copy()
    y_amount = df[TARGET_AMOUNT].astype(float).copy()
    return X, y_adoption, y_amount


def _make_classifier(scale_pos_weight: float) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=30,
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
        n_jobs=-1,
    )


def _make_regressor() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=30,
        verbosity=-1,
        n_jobs=-1,
    )


def _build_pipeline(estimator, feature_columns: Sequence[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(feature_columns)),
            ("model", estimator),
        ]
    )


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _evaluate_classifier(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": _safe_auc(y.to_numpy(), probabilities),
        "pr_auc": _safe_average_precision(y.to_numpy(), probabilities),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y, probabilities)),
    }


def _evaluate_regressor(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    log_predictions = model.predict(X)
    predictions = np.maximum(np.expm1(log_predictions), 0.0)

    rmse = float(np.sqrt(mean_squared_error(y, predictions)))

    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": rmse,
        "r2": float(r2_score(y, predictions)),
    }


def _fit_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationBundle:
    positive_rate = float(y_train.mean())
    scale_pos_weight = (1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0

    logging.info(
        "Entrenando clasificador: positivos train=%.4f | scale_pos_weight=%.4f",
        positive_rate,
        scale_pos_weight,
    )

    model = _build_pipeline(
        _make_classifier(scale_pos_weight=scale_pos_weight),
        feature_columns=X_train.columns,
    )
    model.fit(X_train, y_train)

    train_metrics = _evaluate_classifier(model, X_train, y_train)
    test_metrics = _evaluate_classifier(model, X_test, y_test)

    logging.info(
        "Clasificacion train -> ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Brier=%.4f",
        train_metrics["roc_auc"],
        train_metrics["pr_auc"],
        train_metrics["f1"],
        train_metrics["brier"],
    )
    logging.info(
        "Clasificacion test  -> ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Brier=%.4f",
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
        test_metrics["f1"],
        test_metrics["brier"],
    )

    return EvaluationBundle(model=model, train_metrics=train_metrics, test_metrics=test_metrics)


def _fit_regressor(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
) -> EvaluationBundle:
    logging.info(
        "Entrenando regresor condicional: observaciones train=%s | test=%s",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
    )

    model = _build_pipeline(_make_regressor(), feature_columns=X_train.columns)
    model.fit(X_train, y_train_log)

    train_target = np.expm1(y_train_log)
    test_target = np.expm1(y_test_log)

    train_metrics = _evaluate_regressor(model, X_train, train_target)
    test_metrics = _evaluate_regressor(model, X_test, test_target)

    logging.info(
        "Regresion train -> MAE=%.4f | RMSE=%.4f | R2=%.4f",
        train_metrics["mae"],
        train_metrics["rmse"],
        train_metrics["r2"],
    )
    logging.info(
        "Regresion test  -> MAE=%.4f | RMSE=%.4f | R2=%.4f",
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
    )

    return EvaluationBundle(model=model, train_metrics=train_metrics, test_metrics=test_metrics)


def _fit_final_classifier(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    positive_rate = float(y.mean())
    scale_pos_weight = (1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0
    final_model = _build_pipeline(
        _make_classifier(scale_pos_weight=scale_pos_weight),
        feature_columns=X.columns,
    )
    final_model.fit(X, y)
    return final_model


def _fit_final_regressor(X: pd.DataFrame, y_log: pd.Series) -> Pipeline:
    final_model = _build_pipeline(_make_regressor(), feature_columns=X.columns)
    final_model.fit(X, y_log)
    return final_model


def _score_customers(
    df: pd.DataFrame,
    classifier: Pipeline,
    regressor: Pipeline,
) -> pd.DataFrame:
    adoption_features = df[ADOPTION_FEATURE_COLUMNS].copy()
    amount_features = df[AMOUNT_FEATURE_COLUMNS].copy()

    prob_adopcion = classifier.predict_proba(adoption_features)[:, 1]
    monto_log = regressor.predict(amount_features)
    monto_predicho = np.maximum(np.expm1(monto_log), 0.0)
    valor_esperado_12m = prob_adopcion * monto_predicho

    predictions = pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].values,
            "prob_adopcion": prob_adopcion,
            "monto_predicho": monto_predicho,
            "valor_esperado_12m": valor_esperado_12m,
        }
    )

    decile_rank = predictions["valor_esperado_12m"].rank(method="first", ascending=False)
    try:
        predictions["decel_prioridad"] = pd.qcut(
            decile_rank,
            q=10,
            labels=list(range(10, 0, -1)),
        ).astype(int)
    except ValueError:
        predictions["decel_prioridad"] = pd.cut(
            decile_rank,
            bins=10,
            labels=list(range(10, 0, -1)),
            include_lowest=True,
        ).astype(int)

    return predictions


def _log_decile_distribution(predictions: pd.DataFrame) -> None:
    distribution = (
        predictions.groupby("decel_prioridad", as_index=False)
        .agg(
            clientes=(ID_COLUMN, "count"),
            valor_esperado_promedio=("valor_esperado_12m", "mean"),
            valor_esperado_total=("valor_esperado_12m", "sum"),
        )
        .sort_values("decel_prioridad", ascending=False)
    )

    logging.info("Distribucion de deciles de priorizacion comercial:")
    for _, row in distribution.iterrows():
        logging.info(
            "Decil %s -> clientes=%s | EV promedio=%.4f | EV total=%.4f",
            int(row["decel_prioridad"]),
            f"{int(row['clientes']):,}",
            float(row["valor_esperado_promedio"]),
            float(row["valor_esperado_total"]),
        )


def _save_artifacts(classifier: Pipeline, regressor: Pipeline, predictions: pd.DataFrame) -> None:
    models_dir = _models_dir()
    scores_dir = _scores_path().parent
    models_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = models_dir / "lgbm_adopcion.pkl"
    regressor_path = models_dir / "lgbm_monto.pkl"
    predictions_path = scores_dir / "df_predictions.parquet"

    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)
    predictions.to_parquet(predictions_path, index=False)

    logging.info("Modelo de clasificacion guardado en %s", classifier_path)
    logging.info("Modelo de regresion guardado en %s", regressor_path)
    logging.info("Predicciones guardadas en %s", predictions_path)


def train_pipeline(feature_store: pd.DataFrame) -> TrainingArtifacts:
    """Entrena los modelos, calcula scoring EV y persiste los artefactos finales."""
    _validate_schema(feature_store)
    X_amount, y_adoption, y_amount = _build_xy(feature_store, AMOUNT_FEATURE_COLUMNS)
    X_adoption = feature_store[ADOPTION_FEATURE_COLUMNS].copy()

    logging.info("Feature store cargado: %s filas | %s columnas", f"{len(feature_store):,}", feature_store.shape[1])
    logging.info(
        "Matriz X construida -> adopcion=%s features | monto=%s features",
        len(ADOPTION_FEATURE_COLUMNS),
        len(AMOUNT_FEATURE_COLUMNS),
    )
    logging.info(
        "Distribucion target adopcion -> positivos=%s | negativos=%s",
        f"{int(y_adoption.sum()):,}",
        f"{int((1 - y_adoption).sum()):,}",
    )

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_adoption,
        y_adoption,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_adoption,
    )

    classifier_eval = _fit_classifier(X_train_cls, y_train_cls, X_test_cls, y_test_cls)

    adopted_mask = y_adoption == 1
    X_adopted = X_amount.loc[adopted_mask].copy()
    y_adopted_amount = y_amount.loc[adopted_mask].copy()

    if len(X_adopted) < 2:
        raise ValueError("No hay suficientes registros adoptados para entrenar el modelo de monto.")

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_adopted,
        np.log1p(y_adopted_amount),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    regressor_eval = _fit_regressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg)

    logging.info("Reentrenando modelos finales sobre todo el universo disponible para scoring productivo.")
    final_classifier = _fit_final_classifier(X_adoption, y_adoption)
    final_regressor = _fit_final_regressor(X_adopted, np.log1p(y_adopted_amount))

    predictions = _score_customers(feature_store, final_classifier, final_regressor)
    _log_decile_distribution(predictions)
    _save_artifacts(final_classifier, final_regressor, predictions)

    return TrainingArtifacts(
        classifier_eval=classifier_eval,
        regressor_eval=regressor_eval,
        final_classifier=final_classifier,
        final_regressor=final_regressor,
        predictions=predictions,
    )


def _print_final_report(artifacts: TrainingArtifacts) -> None:
    classifier_test = artifacts.classifier_eval.test_metrics
    regressor_test = artifacts.regressor_eval.test_metrics

    print("\n=== REPORTE FINAL DE ENTRENAMIENTO ===")
    print(
        "Clasificacion test -> "
        f"ROC-AUC={classifier_test['roc_auc']:.4f} | "
        f"PR-AUC={classifier_test['pr_auc']:.4f} | "
        f"F1={classifier_test['f1']:.4f} | "
        f"Brier={classifier_test['brier']:.4f}"
    )
    print(
        "Regresion test     -> "
        f"MAE={regressor_test['mae']:.4f} | "
        f"RMSE={regressor_test['rmse']:.4f} | "
        f"R2={regressor_test['r2']:.4f}"
    )
    print(
        "Predicciones       -> "
        f"filas={len(artifacts.predictions):,} | "
        f"decil_10={int((artifacts.predictions['decel_prioridad'] == 10).sum()):,} | "
        f"decil_1={int((artifacts.predictions['decel_prioridad'] == 1).sum()):,}"
    )


def main() -> TrainingArtifacts:
    feature_store_path = _feature_store_path()
    if not feature_store_path.exists():
        raise FileNotFoundError(
            f"No se encontro el feature store en {feature_store_path}. Ejecuta primero el pipeline de feature engineering."
        )

    logging.info("Cargando feature store desde %s", feature_store_path)
    feature_store = pd.read_parquet(feature_store_path)

    artifacts = train_pipeline(feature_store)
    _print_final_report(artifacts)
    return artifacts


if __name__ == "__main__":
    main()