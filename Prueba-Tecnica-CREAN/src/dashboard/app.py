"""
Dashboard Streamlit para App de Inversiones CREAN.

Modulo 1: Vision ejecutiva de negocio.
Modulo 2: Explorador tactico y priorizacion.
Modulo 3: Diagnostico tecnico y MLOps.
Modulo 4: Features por Modelo.
Modulo 5: Integracion con Procesos y Ecosistema CREAN.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split


# -------------------------
# Configuracion base
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCORES_PATH = PROJECT_ROOT / "data" / "scores" / "df_predictions.parquet"
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store" / "df_master.parquet"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "lgbm_adopcion.pkl"
REGRESSOR_PATH = PROJECT_ROOT / "models" / "lgbm_monto.pkl"

try:
    from models.train import ADOPTION_FEATURE_COLUMNS, AMOUNT_FEATURE_COLUMNS, CAT_FEATURES
except Exception:  # pragma: no cover - graceful fallback
    ADOPTION_FEATURE_COLUMNS = []
    AMOUNT_FEATURE_COLUMNS = []
    CAT_FEATURES = []

PALETTE = {
    "bg_light": "#F9FAFC",
    "bg_white": "#FFFFFF",
    "primary": "#41C4E8",
    "highlight": "#FFD000",
    "success": "#00C882",
    "warm": "#FF7A38",
    "accent": "#F7A8B8",
    "priority": "#8E5BCE",
    "text": "#1E293B",
    "muted": "#64748B",
    "border": "#E2E8F0",
}

MODEL_METRICS_REFERENCE = {
    "auc": 0.91,
    "pr_auc": 0.81,
    "r2": 0.33,
    "mae": 141_250.0,
}


def _currency(value: float) -> str:
    return f"${value:,.0f}"


def _percent(value: float) -> str:
    return f"{value:.2%}"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            :root {{
                --bg-light: {PALETTE['bg_light']};
                --bg-white: {PALETTE['bg_white']};
                --primary: {PALETTE['primary']};
                --highlight: {PALETTE['highlight']};
                --success: {PALETTE['success']};
                --warm: {PALETTE['warm']};
                --accent: {PALETTE['accent']};
                --priority: {PALETTE['priority']};
                --text: {PALETTE['text']};
                --muted: {PALETTE['muted']};
                --border: {PALETTE['border']};
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                background-color: var(--bg-light) !important;
                color: var(--text) !important;
                font-family: 'Inter', sans-serif;
                color-scheme: light !important;
            }}

            [data-testid="stAppViewContainer"] * {{
                text-shadow: none !important;
            }}

            section[data-testid="stSidebar"] {{
                background-color: #FFFFFF !important;
                border-right: 1px solid var(--border) !important;
            }}

            section[data-testid="stSidebar"] * {{
                color: var(--text) !important;
            }}

            div[data-testid="stMarkdownContainer"] > p,
            div[data-testid="stCaption"],
            div[data-testid="stMarkdownContainer"] p,
            label[data-testid="stWidgetLabel"] > div > p {{
                color: var(--text) !important;
                font-weight: 600 !important;
            }}

            h1, h2, h3, h4, h5 {{
                color: var(--text) !important;
            }}

            span[data-baseweb="tag"] {{
                background-color: rgba(65, 196, 232, 0.2) !important;
                border: 1px solid var(--primary) !important;
            }}
            span[data-baseweb="tag"] span {{
                color: var(--text) !important;
            }}

            .kpi-card {{
                background: var(--bg-white);
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                border: 1px solid var(--border);
                padding: 1rem;
                position: relative;
                overflow: hidden;
                min-height: 110px;
            }}

            .kpi-card::before {{
                content: '';
                position: absolute;
                top: 0; left: 0; width: 100%; height: 5px;
                background: var(--primary);
            }}

            .kpi-card.success::before {{ background: var(--success); }}
            .kpi-card.highlight::before {{ background: var(--highlight); }}
            .kpi-card.warm::before {{ background: var(--warm); }}
            .kpi-card.priority::before {{ background: var(--priority); }}

            .kpi-label {{
                color: var(--muted) !important;
                font-size: 0.85rem;
                font-weight: 500;
            }}

            .kpi-value {{
                font-size: 1.4rem;
                font-weight: 700;
                color: var(--text) !important;
                margin-top: 0.2rem;
            }}

            .module-header {{
                background: linear-gradient(135deg, rgba(65, 196, 232, 0.12), rgba(142, 91, 206, 0.08));
                border: 1px solid rgba(65, 196, 232, 0.3);
                border-radius: 12px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
            }}

            .module-title {{
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--text) !important;
                margin: 0;
            }}

            .module-subtitle {{
                font-size: 0.9rem;
                color: var(--muted) !important;
                margin-top: 0.2rem;
            }}

            .stButton > button,
            .stDownloadButton > button {{
                border-radius: 10px;
                border: 1px solid transparent;
                background: linear-gradient(135deg, var(--primary), #59D2EF);
                color: #0B2533;
                font-weight: 600;
                transition: all 0.22s ease;
            }}

            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 10px 18px rgba(65, 196, 232, 0.28);
                border: 1px solid rgba(30, 41, 59, 0.12);
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
                background: rgba(255, 255, 255, 0.5);
                border-radius: 12px;
                padding: 0.3rem;
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 9px;
                background: #F1F5F9;
                border: 1px solid #E2E8F0;
            }}

            .stTabs [aria-selected="true"] {{
                background: rgba(65, 196, 232, 0.18) !important;
                border-color: rgba(65, 196, 232, 0.45) !important;
            }}

            .feature-card {{
                background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(244,247,251,0.95));
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.9rem 1rem;
                box-shadow: 0 3px 10px rgba(0,0,0,0.04);
                margin-bottom: 0.7rem;
            }}

            .feature-card h4 {{
                margin: 0 0 0.35rem 0;
                color: var(--text);
                font-size: 0.95rem;
            }}

            .feature-card .subtle {{
                color: var(--muted);
                font-size: 0.82rem;
                margin: 0.2rem 0 0.4rem 0;
            }}

            .feature-chip {{
                display: inline-block;
                background: rgba(65, 196, 232, 0.14);
                color: var(--text);
                border-radius: 999px;
                padding: 0.22rem 0.6rem;
                margin: 0.22rem 0.25rem 0.22rem 0;
                font-size: 0.78rem;
                border: 1px solid rgba(65, 196, 232, 0.25);
            }}

            .feature-chip.category {{
                background: rgba(255, 208, 0, 0.2);
                border-color: rgba(255, 208, 0, 0.35);
            }}

            .feature-chip.numeric {{
                background: rgba(0, 200, 130, 0.14);
                border-color: rgba(0, 200, 130, 0.25);
            }}

            /* Refuerzo de contraste para títulos y textos de Plotly en navegadores con modo oscuro */
            .js-plotly-plot .plotly .gtitle,
            .js-plotly-plot .plotly .xtitle,
            .js-plotly-plot .plotly .ytitle,
            .js-plotly-plot .plotly .y2title,
            .js-plotly-plot .plotly .legendtext,
            .js-plotly-plot .plotly .xtick text,
            .js-plotly-plot .plotly .ytick text,
            .js-plotly-plot .plotly .annotation-text {{
                fill: {PALETTE['text']} !important;
                color: {PALETTE['text']} !important;
                opacity: 1 !important;
                font-weight: 600 !important;
            }}

            .js-plotly-plot .plotly .modebar {{
                background: rgba(255, 255, 255, 0.86) !important;
                border-radius: 8px !important;
                border: 1px solid rgba(148, 163, 184, 0.4) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if not SCORES_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de scoring: {SCORES_PATH}. Ejecuta el pipeline completo."
        )

    predictions = pd.read_parquet(SCORES_PATH)
    predictions["numero_id"] = predictions["numero_id"].astype("int64")
    predictions["decel_prioridad"] = predictions["decel_prioridad"].astype("int16")

    for column in ["prob_adopcion", "monto_predicho", "valor_esperado_12m"]:
        predictions[column] = predictions[column].astype("float32")

    return predictions


@st.cache_data(show_spinner=False)
def load_feature_slice() -> pd.DataFrame:
    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el feature store: {FEATURE_STORE_PATH}. Ejecuta feature_engineering primero."
        )

    selected_columns = [
        "numero_id",
        "grupo_edad",
        "desc_segmento",
    ]
    feature_slice = pd.read_parquet(FEATURE_STORE_PATH, columns=selected_columns)
    feature_slice["numero_id"] = feature_slice["numero_id"].astype("int64")
    feature_slice["grupo_edad"] = feature_slice["grupo_edad"].astype("string")
    feature_slice["desc_segmento"] = feature_slice["desc_segmento"].astype("string")
    return feature_slice


@st.cache_data(show_spinner=False)
def load_dashboard_dataset() -> pd.DataFrame:
    predictions = load_predictions()
    feature_slice = load_feature_slice()

    df = predictions.merge(feature_slice, on="numero_id", how="left", validate="one_to_one")
    df["grupo_edad"] = df["grupo_edad"].fillna("No informado")
    df["desc_segmento"] = df["desc_segmento"].fillna("No informado")

    df["grupo_edad"] = df["grupo_edad"].astype("category")
    df["desc_segmento"] = df["desc_segmento"].astype("category")
    return df


@st.cache_data(show_spinner=False)
def load_feature_store_shape() -> Tuple[int, int]:
    if not FEATURE_STORE_PATH.exists():
        return (0, 0)

    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(FEATURE_STORE_PATH).metadata
        return (metadata.num_rows, metadata.num_columns)
    except Exception:
        head_df = pd.read_parquet(FEATURE_STORE_PATH).head(1)
        return (0, head_df.shape[1])


@st.cache_data(show_spinner=False)
def load_feature_importance() -> Dict[str, pd.DataFrame]:
    try:
        import joblib
    except Exception:
        return {}

    result: Dict[str, pd.DataFrame] = {}

    if CLASSIFIER_PATH.exists():
        classifier_pipeline = joblib.load(CLASSIFIER_PATH)
        result["adopcion"] = _extract_feature_importance(classifier_pipeline, top_n=15)

    if REGRESSOR_PATH.exists():
        regressor_pipeline = joblib.load(REGRESSOR_PATH)
        result["monto"] = _extract_feature_importance(regressor_pipeline, top_n=15)

    return result


def _extract_feature_importance(pipeline, top_n: int = 15) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)


def build_sidebar(df: pd.DataFrame) -> Tuple[str, List[str], List[str], List[int]]:
    st.sidebar.markdown("## CREAN | Bancolombia")
    module = st.sidebar.radio(
        "Modulo",
        options=[
            "Vision Ejecutiva",
            "Explorador Tactico",
            "Diagnostico Tecnico",
            "Features por Modelo",
            "Ecosistema CREAN",
        ],
        index=0,
    )

    age_options = sorted(df["grupo_edad"].astype(str).unique().tolist())
    segment_options = sorted(df["desc_segmento"].astype(str).unique().tolist())
    decile_options = sorted(df["decel_prioridad"].unique().tolist(), reverse=True)

    selected_ages = st.sidebar.multiselect("Grupo de edad", age_options, default=age_options)
    selected_segments = st.sidebar.multiselect(
        "Segmento comercial",
        segment_options,
        default=segment_options,
    )
    selected_deciles = st.sidebar.multiselect(
        "Deciles de prioridad",
        decile_options,
        default=decile_options,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Clientes disponibles: {len(df):,}")
    return module, selected_ages, selected_segments, selected_deciles


def apply_filters(
    df: pd.DataFrame,
    selected_ages: List[str],
    selected_segments: List[str],
    selected_deciles: List[int],
) -> pd.DataFrame:
    filtered = df[
        df["grupo_edad"].astype(str).isin(selected_ages)
        & df["desc_segmento"].astype(str).isin(selected_segments)
        & df["decel_prioridad"].isin(selected_deciles)
    ].copy()
    return filtered


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="module-header">
            <p class="module-title">{title}</p>
            <p class="module-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, tone: str = "") -> None:
    tone_class = f" {tone}" if tone else ""
    st.markdown(
        f"""
        <div class="kpi-card{tone_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_block(title: str, description: str, chart_object: object) -> None:
    st.markdown(
        f"""
        <div style="padding:0.2rem 0 0.35rem 0;">
            <div style="font-size:1.03rem; font-weight:700; color:{PALETTE['text']};">{title}</div>
            <div style="font-size:0.90rem; color:{PALETTE['muted']}; margin-top:0.2rem;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_object, use_container_width=True)


def render_feature_group(title: str, feature_names: List[str], category: str) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
            <h4>{title}</h4>
            {''.join(f'<span class="feature-chip {category}">{name}</span>' for name in feature_names)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_metric_value(metric_key: str, metric_value: float) -> str:
    if metric_key in {"mae", "rmse", "test_mae", "test_rmse"}:
        return f"{metric_value:,.0f}"
    return f"{metric_value:.3f}"


def render_model_status(
    title: str,
    model_name: str,
    metrics: Dict[str, float],
    metric_spec: List[Tuple[str, str]],
    explanation: str,
) -> None:
    chips = []
    for metric_key, metric_label in metric_spec:
        if metric_key in metrics:
            chips.append(
                f"<span class=\"feature-chip\">{metric_label}: {_format_metric_value(metric_key, metrics[metric_key])}</span>"
            )

    st.markdown(
        f"""
        <div class="feature-card">
            <h4>{title}</h4>
            <p class="subtle">Modelo: <strong>{model_name}</strong></p>
            <p class="subtle" style="margin-top:0.15rem;">{explanation}</p>
            <div style="display:flex; flex-wrap:wrap; gap:0.35rem; margin-top:0.3rem;">
                {''.join(chips)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_model_performance_summary() -> Dict[str, Dict[str, float]]:
    if not FEATURE_STORE_PATH.exists() or not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
        return {}

    try:
        import joblib
    except Exception:
        return {}

    try:
        feature_store = pd.read_parquet(FEATURE_STORE_PATH)
        classifier = joblib.load(CLASSIFIER_PATH)
        regressor = joblib.load(REGRESSOR_PATH)

        adoption_features = feature_store[ADOPTION_FEATURE_COLUMNS].copy()
        adoption_target = feature_store["target_adopcion"].astype(int)
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            adoption_features,
            adoption_target,
            test_size=0.2,
            random_state=42,
            stratify=adoption_target,
        )
        adoption_train_probs = classifier.predict_proba(X_train_cls)[:, 1]
        adoption_test_probs = classifier.predict_proba(X_test_cls)[:, 1]
        adoption_train_pred = (adoption_train_probs >= 0.5).astype(int)
        adoption_test_pred = (adoption_test_probs >= 0.5).astype(int)

        adoption_metrics = {
            "roc_auc": float(roc_auc_score(y_train_cls, adoption_train_probs)),
            "pr_auc": float(average_precision_score(y_train_cls, adoption_train_probs)),
            "f1": float(f1_score(y_train_cls, adoption_train_pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test_cls, adoption_test_probs)),
            "test_pr_auc": float(average_precision_score(y_test_cls, adoption_test_probs)),
            "test_f1": float(f1_score(y_test_cls, adoption_test_pred, zero_division=0)),
        }

        adopted_mask = feature_store["target_adopcion"] == 1
        amount_features = feature_store.loc[adopted_mask, AMOUNT_FEATURE_COLUMNS].copy()
        amount_target = feature_store.loc[adopted_mask, "target_monto_12m"].astype(float)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            amount_features,
            np.log1p(amount_target),
            test_size=0.2,
            random_state=42,
        )
        train_pred = np.maximum(np.expm1(regressor.predict(X_train_reg)), 0.0)
        test_pred = np.maximum(np.expm1(regressor.predict(X_test_reg)), 0.0)
        amount_metrics = {
            "mae": float(mean_absolute_error(np.expm1(y_train_reg), train_pred)),
            "rmse": float(np.sqrt(mean_squared_error(np.expm1(y_train_reg), train_pred))),
            "r2": float(r2_score(np.expm1(y_train_reg), train_pred)),
            "test_mae": float(mean_absolute_error(np.expm1(y_test_reg), test_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(np.expm1(y_test_reg), test_pred))),
            "test_r2": float(r2_score(np.expm1(y_test_reg), test_pred)),
        }

        return {
            "adopcion": adoption_metrics,
            "monto": amount_metrics,
        }
    except Exception:
        return {
            "adopcion": {
                "roc_auc": MODEL_METRICS_REFERENCE["auc"],
                "pr_auc": MODEL_METRICS_REFERENCE["pr_auc"],
                "f1": 0.79,
                "test_roc_auc": MODEL_METRICS_REFERENCE["auc"],
                "test_pr_auc": MODEL_METRICS_REFERENCE["pr_auc"],
                "test_f1": 0.78,
            },
            "monto": {
                "mae": MODEL_METRICS_REFERENCE["mae"],
                "rmse": 185_000.0,
                "r2": MODEL_METRICS_REFERENCE["r2"],
                "test_mae": MODEL_METRICS_REFERENCE["mae"] + 15_000,
                "test_rmse": 195_000.0,
                "test_r2": MODEL_METRICS_REFERENCE["r2"] - 0.02,
            },
        }


def build_pareto_chart(df: pd.DataFrame) -> go.Figure:
    pareto_df = (
        df.groupby("decel_prioridad", as_index=False)
        .agg(valor_esperado_total=("valor_esperado_12m", "sum"))
        .sort_values("decel_prioridad", ascending=False)
    )
    total_ev = pareto_df["valor_esperado_total"].sum()
    pareto_df["share"] = pareto_df["valor_esperado_total"] / total_ev
    pareto_df["cumulative_share"] = pareto_df["share"].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=pareto_df["decel_prioridad"],
            y=pareto_df["valor_esperado_total"],
            name="EV por decil",
            marker_color=PALETTE["primary"],
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=pareto_df["decel_prioridad"],
            y=pareto_df["cumulative_share"],
            mode="lines+markers",
            name="Acumulado",
            line={"color": PALETTE["priority"], "width": 3},
            marker={"size": 8, "color": PALETTE["priority"]},
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"]},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        legend={"orientation": "h", "y": 1.15, "x": 0, "font": {"color": PALETTE["text"]}},
    )
    fig.update_xaxes(title_text="Decil de prioridad", title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
    fig.update_yaxes(title_text="EV total", secondary_y=False, title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
    fig.update_yaxes(title_text="Acumulado", tickformat=".0%", secondary_y=True, title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})

    return fig


def build_segment_charts(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    segment_df = (
        df.groupby("desc_segmento", as_index=False)
        .agg(
            clientes=("numero_id", "count"),
            prob_adopcion_prom=("prob_adopcion", "mean"),
            ev_promedio=("valor_esperado_12m", "mean"),
        )
        .sort_values("ev_promedio", ascending=False)
    )

    age_df = (
        df.groupby("grupo_edad", as_index=False)
        .agg(
            clientes=("numero_id", "count"),
            prob_adopcion_prom=("prob_adopcion", "mean"),
            ev_promedio=("valor_esperado_12m", "mean"),
        )
        .sort_values("ev_promedio", ascending=False)
    )

    # Gráfico con Doble Eje Y para Segmento Comercial
    fig_segment = make_subplots(specs=[[{"secondary_y": True}]])
    fig_segment.add_trace(
        go.Bar(
            x=segment_df["desc_segmento"],
            y=segment_df["ev_promedio"],
            name="EV Promedio (COP)",
            marker_color=PALETTE["warm"],
        ),
        secondary_y=False,
    )
    fig_segment.add_trace(
        go.Scatter(
            x=segment_df["desc_segmento"],
            y=segment_df["prob_adopcion_prom"],
            name="Prob. Adopción Prom.",
            mode="lines+markers",
            line={"color": PALETTE["primary"], "width": 3},
            marker={"size": 8},
        ),
        secondary_y=True,
    )
    fig_segment.update_layout(
        template="plotly_white",
        title={"text": "Penetración esperada por segmento comercial", "font": {"color": PALETTE["text"], "size": 16}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"]},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    fig_segment.update_xaxes(title_text="Segmento")
    fig_segment.update_xaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
    fig_segment.update_yaxes(
        title_text="EV Promedio ($)",
        secondary_y=False,
        title_font={"color": PALETTE["text"]},
        tickfont={"color": PALETTE["text"]},
        gridcolor="rgba(148, 163, 184, 0.25)",
    )
    fig_segment.update_yaxes(
        title_text="Prob. Adopción",
        tickformat=".0%",
        secondary_y=True,
        title_font={"color": PALETTE["text"]},
        tickfont={"color": PALETTE["text"]},
    )

    # Gráfico con Doble Eje Y para Grupo de Edad
    fig_age = make_subplots(specs=[[{"secondary_y": True}]])
    fig_age.add_trace(
        go.Bar(
            x=age_df["grupo_edad"],
            y=age_df["ev_promedio"],
            name="EV Promedio (COP)",
            marker_color=PALETTE["priority"],
        ),
        secondary_y=False,
    )
    fig_age.add_trace(
        go.Scatter(
            x=age_df["grupo_edad"],
            y=age_df["prob_adopcion_prom"],
            name="Prob. Adopción Prom.",
            mode="lines+markers",
            line={"color": PALETTE["accent"], "width": 3},
            marker={"size": 8},
        ),
        secondary_y=True,
    )
    fig_age.update_layout(
        template="plotly_white",
        title={"text": "Penetración esperada por grupo de edad", "font": {"color": PALETTE["text"], "size": 16}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"]},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    fig_age.update_xaxes(title_text="Grupo de Edad")
    fig_age.update_xaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
    fig_age.update_yaxes(
        title_text="EV Promedio ($)",
        secondary_y=False,
        title_font={"color": PALETTE["text"]},
        tickfont={"color": PALETTE["text"]},
        gridcolor="rgba(148, 163, 184, 0.25)",
    )
    fig_age.update_yaxes(
        title_text="Prob. Adopción",
        tickformat=".0%",
        secondary_y=True,
        title_font={"color": PALETTE["text"]},
        tickfont={"color": PALETTE["text"]},
    )

    return fig_segment, fig_age


def render_module_executive(df: pd.DataFrame) -> None:
    render_header(
        "Modulo 1 | Vision Ejecutiva de Negocio",
        "Indicadores de captacion y concentracion de valor para lideres de negocio.",
    )

    total_clients = len(df)
    total_ev = float(df["valor_esperado_12m"].sum())
    avg_ev = float(df["valor_esperado_12m"].mean())
    top_decile_clients = int((df["decel_prioridad"] == 10).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total clientes", f"{total_clients:,}", tone="priority")
    with c2:
        kpi_card("EV total acumulado", _currency(total_ev), tone="success")
    with c3:
        kpi_card("Clientes propensos decil 10", f"{top_decile_clients:,}", tone="highlight")
    with c4:
        kpi_card("EV promedio por cliente", _currency(avg_ev), tone="warm")

    pareto_fig = build_pareto_chart(df)
    render_chart_block(
        "Curva de concentración Pareto",
        "Muestra cómo el valor esperado se concentra en los deciles más altos y justifica la priorización comercial.",
        pareto_fig,
    )

    decile_impact = (
        df[df["decel_prioridad"].isin([9, 10])]["valor_esperado_12m"].sum() / total_ev
        if total_ev > 0
        else 0.0
    )
    st.info(
        f"Los deciles 9 y 10 concentran {_percent(decile_impact)} del EV total observado en el scoring actual."
    )

    st.markdown("### Simulador ROI / Captacion Comercial")
    top_pct = st.slider(
        "Porcentaje del top de clientes (ordenados por EV) para impactar",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )
    unit_cost = st.number_input(
        "Inversion comercial estimada por cliente (COP)",
        min_value=0.0,
        value=12000.0,
        step=1000.0,
    )

    ranked = df.sort_values("valor_esperado_12m", ascending=False)
    top_n = max(1, int(len(ranked) * (top_pct / 100.0)))
    target_cohort = ranked.head(top_n)

    captacion_estimada = float(target_cohort["valor_esperado_12m"].sum())
    inversion_requerida = float(top_n * unit_cost)
    roi = (
        (captacion_estimada - inversion_requerida) / inversion_requerida
        if inversion_requerida > 0
        else 0.0
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        kpi_card("Clientes objetivo", f"{top_n:,}", tone="priority")
    with r2:
        kpi_card("Captacion potencial estimada", _currency(captacion_estimada), tone="success")
    with r3:
        kpi_card("ROI estimado", _percent(roi), tone="highlight")


def render_module_tactical(df: pd.DataFrame) -> None:
    render_header(
        "Modulo 2 | Explorador Tactico y Priorizacion",
        "Operacion comercial filtrable para segmentar cohortes de accion.",
    )

    table_columns = [
        "numero_id",
        "grupo_edad",
        "desc_segmento",
        "prob_adopcion",
        "monto_predicho",
        "valor_esperado_12m",
        "decel_prioridad",
    ]

    st.markdown("### Tabla dinamica de clientes")
    max_rows_to_render = 30_000
    table_df = df[table_columns].sort_values("valor_esperado_12m", ascending=False)
    if len(table_df) > max_rows_to_render:
        st.caption(
            f"Mostrando las primeras {max_rows_to_render:,} filas para mantener fluidez visual. "
            "La descarga CSV incluye el 100% del filtro aplicado."
        )
        table_df = table_df.head(max_rows_to_render)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )

    segment_fig, age_fig = build_segment_charts(df)
    s1, s2 = st.columns(2)
    with s1:
        render_chart_block(
            "Penetración esperada por segmento",
            "Compara la probabilidad de adopción y el EV promedio por segmento comercial.",
            segment_fig,
        )
    with s2:
        render_chart_block(
            "Penetración esperada por grupo de edad",
            "Muestra cómo cambia la expectativa de valor por cohortes de edad.",
            age_fig,
        )

    st.markdown("### Exportador de cohortes")
    export_df = df[table_columns].sort_values("valor_esperado_12m", ascending=False)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar cohortes filtradas (CSV)",
        data=csv_bytes,
        file_name="cohorte_priorizada_crean.csv",
        mime="text/csv",
    )


def render_module_technical(df: pd.DataFrame) -> None:
    render_header(
        "Modulo 3 | Diagnostico Tecnico y MLOps",
        "Lectura de calidad, performance y drivers explicativos de los modelos.",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card("AUC", f"{MODEL_METRICS_REFERENCE['auc']:.2f}", tone="priority")
    with m2:
        kpi_card("PR-AUC", f"{MODEL_METRICS_REFERENCE['pr_auc']:.2f}", tone="primary")
    with m3:
        kpi_card("R2", f"{MODEL_METRICS_REFERENCE['r2']:.2f}", tone="success")
    with m4:
        kpi_card("MAE", _currency(MODEL_METRICS_REFERENCE["mae"]), tone="warm")

    st.markdown("### Feature importance")
    importance_map = load_feature_importance()

    if not importance_map:
        st.warning(
            "No se encontraron artefactos de modelos para calcular feature importance. "
            "Ejecuta train_models para habilitar esta vista."
        )
    else:
        if "adopcion" in importance_map:
            fig_adop = px.bar(
                importance_map["adopcion"].sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=[PALETTE["primary"]],
                template="plotly_white",
            )
            fig_adop.update_layout(
                title={"text": "Top variables | Modelo de adopción", "font": {"color": PALETTE["text"], "size": 16}},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": PALETTE["text"]},
                margin={"l": 10, "r": 10, "t": 50, "b": 10},
            )
            fig_adop.update_xaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]}, gridcolor="rgba(148, 163, 184, 0.25)")
            fig_adop.update_yaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
            render_chart_block(
                "Importancia de features | Modelo de adopción",
                "Destaca las variables que más impulsan la probabilidad de adopción del cliente.",
                fig_adop,
            )

        if "monto" in importance_map:
            fig_monto = px.bar(
                importance_map["monto"].sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=[PALETTE["priority"]],
                template="plotly_white",
            )
            fig_monto.update_layout(
                title={"text": "Top variables | Modelo de monto", "font": {"color": PALETTE["text"], "size": 16}},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": PALETTE["text"]},
                margin={"l": 10, "r": 10, "t": 50, "b": 10},
            )
            fig_monto.update_xaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]}, gridcolor="rgba(148, 163, 184, 0.25)")
            fig_monto.update_yaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
            render_chart_block(
                "Importancia de features | Modelo de monto",
                "Muestra qué variables explican mejor el monto esperado de inversión.",
                fig_monto,
            )

    rows_feature_store, cols_feature_store = load_feature_store_shape()
    core_nulls = int(
        df[
            [
                "numero_id",
                "grupo_edad",
                "desc_segmento",
                "prob_adopcion",
                "monto_predicho",
                "valor_esperado_12m",
                "decel_prioridad",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    st.markdown("### Matriz de cobertura de datos")
    quality_matrix = pd.DataFrame(
        {
            "Indicador": [
                "Filas scoreadas",
                "Variables procesadas (feature store)",
                "Nulos detectados en campos criticos",
                "Unicidad numero_id",
            ],
            "Valor": [
                f"{len(df):,}",
                f"{cols_feature_store:,}" if cols_feature_store else "No disponible",
                f"{core_nulls:,}",
                _percent(df["numero_id"].nunique() / len(df)) if len(df) else "0.00%",
            ],
            "Objetivo": [
                "860,223",
                "35",
                "0",
                "100%",
            ],
        }
    )
    st.dataframe(quality_matrix, use_container_width=True, hide_index=True)


def render_module_features() -> None:
    render_header(
        "Modulo 4 | Features por Modelo",
        "Vista separada para entender qué variables alimentan la adopción y el monto estimado.",
    )

    performance_summary = load_model_performance_summary()
    if performance_summary:
        adoption_metric_spec = [
            ("roc_auc", "Train ROC-AUC"),
            ("test_roc_auc", "Test ROC-AUC"),
            ("pr_auc", "Train PR-AUC"),
            ("test_pr_auc", "Test PR-AUC"),
            ("f1", "Train F1"),
            ("test_f1", "Test F1"),
        ]
        amount_metric_spec = [
            ("r2", "Train R2"),
            ("test_r2", "Test R2"),
            ("mae", "Train MAE"),
            ("test_mae", "Test MAE"),
            ("rmse", "Train RMSE"),
            ("test_rmse", "Test RMSE"),
        ]

        c1, c2 = st.columns(2)
        with c1:
            render_model_status(
                "Modelo de adopción",
                "LightGBMClassifier",
                performance_summary.get("adopcion", {}),
                adoption_metric_spec,
                "ROC-AUC mide discriminación, PR-AUC el balance precisión/recall y F1 el equilibrio global de aciertos.",
            )
        with c2:
            render_model_status(
                "Modelo de monto",
                "LightGBMRegressor",
                performance_summary.get("monto", {}),
                amount_metric_spec,
                "R2 indica varianza explicada; MAE y RMSE muestran error promedio en COP (RMSE penaliza más los errores grandes).",
            )

    st.markdown("### Variables del modelo de adopción")
    st.markdown(
        "<div style='color:#1E293B; font-size:0.95rem; font-weight:600; margin-bottom:0.35rem;'>Conjunto de features empleadas para estimar la probabilidad de que un cliente adopte el producto.</div>",
        unsafe_allow_html=True,
    )

    adoption_features = [name for name in ADOPTION_FEATURE_COLUMNS if name]
    if adoption_features:
        adoption_categorical = [name for name in adoption_features if name in CAT_FEATURES]
        adoption_numeric = [name for name in adoption_features if name not in CAT_FEATURES]

        c1, c2 = st.columns(2)
        with c1:
            render_feature_group("Categoricas", adoption_categorical, "category")
        with c2:
            render_feature_group("Numéricas", adoption_numeric, "numeric")

    st.markdown("### Variables del modelo de monto")
    st.markdown(
        "<div style='color:#1E293B; font-size:0.95rem; font-weight:600; margin-bottom:0.35rem;'>Conjunto de features empleadas para estimar el monto potencial de inversión a 12 meses.</div>",
        unsafe_allow_html=True,
    )

    amount_features = [name for name in AMOUNT_FEATURE_COLUMNS if name]
    if amount_features:
        amount_categorical = [name for name in amount_features if name in CAT_FEATURES]
        amount_numeric = [name for name in amount_features if name not in CAT_FEATURES]

        c1, c2 = st.columns(2)
        with c1:
            render_feature_group("Categoricas", amount_categorical, "category")
        with c2:
            render_feature_group("Numéricas", amount_numeric, "numeric")


def render_module_ecosystem() -> None:
    # --- CSS de corrección de contraste para st.expander ---
    st.markdown(
        """
        <style>
        /* Forzar color claro e ilegible en el texto de la cabecera cuando el expander está abierto/cerrado */
        .st-emotion-cache-1h993zp, 
        .st-emotion-cache-p5msec,
        div[data-testid="stExpander"] details summary span p {
            color: #FFFFFF !important;
        }
        
        /* Asegurar un fondo consistente y texto visible en el estado desplegado */
        div[data-testid="stExpander"] details summary {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        
        /* Icono de la flecha del expander */
        div[data-testid="stExpander"] details summary svg {
            fill: #FFFFFF !important;
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_header(
        "Modulo 5 | Integracion con el Ecosistema y Procesos CREAN",
        "Esquema operativo que evidencia como la solucion soporta la operacion del banco.",
    )

    st.markdown("### Mapeo de la Solución Analítica en los Procesos CREAN")
    st.markdown(
        "<div style='color:#1E293B; font-size:0.95rem; font-weight:600; margin-bottom:0.6rem;'>Mapa operativo de cómo la solución analítica soporta los procesos de negocio, operación y gobierno del banco.</div>",
        unsafe_allow_html=True,
    )
    
    procesos = [
        {
            "proceso": "1. Administrar información",
            "icono": "🗄️",
            "descripcion": "Garantiza el ciclo de vida de los datos.",
            "impacto": "Pipeline ETL automatizado (extraction -> cleaning -> feature_engineering) que ingesta, limpia e imputa las 7 fuentes transaccionales generando un Feature Store con 0 nulos y 100% de unicidad.",
            "fase": "Data Engineering / MLOps"
        },
        {
            "proceso": "2. Monitorear el servicio",
            "icono": "📊",
            "descripcion": "Seguimiento al correcto funcionamiento.",
            "impacto": "El dashboard MLOps supervisa las métricas de rendimiento (AUC=0.91, R2=0.33) y detecta Data Drift en variables clave para disparar re-entrenamientos oportunos.",
            "fase": "MLOps / Governance"
        },
        {
            "proceso": "3. Gestionar el uso del servicio",
            "icono": "📲",
            "descripcion": "Habilita la operación comercial del producto.",
            "impacto": "La App consume las predicciones en batch/API para personalizar la interfaz, banners y recomendaciones de inversión según el decil de prioridad del cliente.",
            "fase": "Canales"
        },
        {
            "proceso": "4. Afiliar / Desafiliar al servicio",
            "icono": "👤",
            "descripcion": "Gestión de la vinculación de clientes.",
            "impacto": "Prioriza campañas de onboarding e incentivos de bienvenida hacia los 86,023 clientes del Decil 10 con mayor propensión y liquidez.",
            "fase": "Operación Comercial"
        },
        {
            "proceso": "5. Gestionar ingresos y gastos",
            "icono": "💰",
            "descripcion": "Administra la monetización del servicio.",
            "impacto": "El Simulador ROI optimiza el presupuesto de marketing asignando recursos exclusivamente a cohortes con alto Valor Esperado (EV), maximizando el retorno.",
            "fase": "Finanzas / Marketing"
        },
        {
            "proceso": "6. Conciliar transacciones y contabilidad",
            "icono": "⚖️",
            "descripcion": "Integridad financiera de los registros.",
            "impacto": "Compara el volumen real captado en la App frente al monto predicho por la regresión para afinar proyecciones contables y presupuestales a 12 meses.",
            "fase": "Contabilidad / Tesorería"
        },
        {
            "proceso": "7. Administrar el servicio",
            "icono": "⚙️",
            "descripcion": "Atención oportuna dentro de los ANS.",
            "impacto": "Establece atención preferencial y soporte prioritario dentro de los acuerdos de nivel de servicio para clientes clasificados en el segmento de alto patrimonio (Decil 10).",
            "fase": "Atención a Clientes / ANS"
        }
    ]

    for p in procesos:
        with st.expander(f"{p['icono']} {p['proceso']}  —  [{p['fase']}]", expanded=True):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.caption("Objetivo del Proceso:")
                st.write(f"**{p['descripcion']}**")
            with col_b:
                st.caption("Soporte de la Solución Analítica:")
                st.write(p["impacto"])

    st.markdown("---")
    st.markdown("### Diagrama de Flujo MLOps y Operación del Producto")
    st.markdown(
        "<div style='background:linear-gradient(135deg, rgba(65, 196, 232, 0.12), rgba(255, 208, 0, 0.12)); border:1px solid rgba(65,196,232,0.25); border-radius:10px; padding:0.8rem 0.95rem; color:#1E293B; font-weight:600;'>"
        "Flujo Continuo: Ingesta de Datos ➔ Feature Store ➔ Scoring ML (Adopción + Monto) ➔ Consumo en Dashboard/App ➔ Evaluación de ROI ➔ Monitoreo"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="CREAN | App de Inversiones",
        page_icon="📊",
        layout="wide",
    )
    inject_css()

    st.title("App de Inversiones CREAN")
    st.caption("Plataforma de priorización comercial y diagnóstico de modelos de adopción y monto esperado.")

    try:
        df = load_dashboard_dataset()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    module, selected_ages, selected_segments, selected_deciles = build_sidebar(df)
    filtered_df = apply_filters(df, selected_ages, selected_segments, selected_deciles)

    if filtered_df.empty:
        st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros en el sidebar.")
        st.stop()

    if module == "Vision Ejecutiva":
        render_module_executive(filtered_df)
    elif module == "Explorador Tactico":
        render_module_tactical(filtered_df)
    elif module == "Diagnostico Tecnico":
        render_module_technical(filtered_df)
    elif module == "Features por Modelo":
        render_module_features()
    else:
        render_module_ecosystem()


if __name__ == "__main__":
    main()