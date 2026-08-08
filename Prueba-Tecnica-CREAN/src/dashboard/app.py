"""
Dashboard Streamlit para App de Inversiones CREAN.

Modulo 1: Vision ejecutiva de negocio.
Modulo 2: Explorador tactico y priorizacion.
Modulo 3: Diagnostico tecnico y MLOps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# -------------------------
# Configuracion base
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = PROJECT_ROOT / "data" / "scores" / "df_predictions.parquet"
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store" / "df_master.parquet"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "lgbm_adopcion.pkl"
REGRESSOR_PATH = PROJECT_ROOT / "models" / "lgbm_monto.pkl"

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

            /* 1. Forzar variables de modo claro en la raíz */
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

            /* Forzar fondo general claro */
            html, body, [data-testid="stAppViewContainer"] {{
                background-color: var(--bg-light) !important;
                color: var(--text) !important;
                font-family: 'Inter', sans-serif;
            }}

            /* 2. Forzar estilizado de la barra lateral (Sidebar) */
            section[data-testid="stSidebar"] {{
                background-color: #FFFFFF !important;
                border-right: 1px solid var(--border) !important;
            }}

            section[data-testid="stSidebar"] * {{
                color: var(--text) !important;
            }}

            /* Estilizado de radio buttons y labels en el sidebar */
            div[data-testid="stMarkdownContainer"] > p,
            label[data-testid="stWidgetLabel"] > div > p {{
                color: var(--text) !important;
                font-weight: 600 !important;
            }}

            /* 3. Estilizado de etiquetas Multiselect */
            span[data-baseweb="tag"] {{
                background-color: rgba(65, 196, 232, 0.2) !important;
                border: 1px solid var(--primary) !important;
            }}
            span[data-baseweb="tag"] span {{
                color: var(--text) !important;
            }}

            /* 4. Tarjetas KPI */
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

            /* Header del Módulo */
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
    st.sidebar.markdown("## CREAN | Navegacion")
    module = st.sidebar.radio(
        "Modulo",
        options=[
            "Vision Ejecutiva",
            "Explorador Tactico",
            "Diagnostico Tecnico",
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
        title={"text": "Curva de Concentración Pareto | Valor Esperado por Decil", "font": {"color": PALETTE["text"]}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"]},
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"color": PALETTE["text"]}},
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

    fig_segment = px.bar(
        segment_df,
        x="desc_segmento",
        y=["prob_adopcion_prom", "ev_promedio"],
        barmode="group",
        title="Penetracion esperada por segmento comercial",
        color_discrete_sequence=[PALETTE["primary"], PALETTE["warm"]],
        template="plotly_white",
    )
    fig_segment.update_layout(
        xaxis_title="Segmento",
        yaxis_title="Valor",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"]},
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        legend={"font": {"color": PALETTE["text"]}},
    )
    fig_segment.update_xaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
    fig_segment.update_yaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})

    fig_age = px.bar(
        age_df,
        x="grupo_edad",
        y=["prob_adopcion_prom", "ev_promedio"],
        barmode="group",
        title="Penetracion esperada por grupo de edad",
        color_discrete_sequence=[PALETTE["accent"], PALETTE["priority"]],
        template="plotly_white",
    )
    fig_age.update_layout(
        xaxis_title="Grupo de edad",
        yaxis_title="Valor",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"]},
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        legend={"font": {"color": PALETTE["text"]}},
    )
    fig_age.update_xaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
    fig_age.update_yaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})

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
    st.plotly_chart(pareto_fig, use_container_width=True)

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
        st.plotly_chart(segment_fig, use_container_width=True)
    with s2:
        st.plotly_chart(age_fig, use_container_width=True)

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
                title="Top variables | Modelo de adopcion",
                color_discrete_sequence=[PALETTE["primary"]],
                template="plotly_white",
            )
            fig_adop.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": PALETTE["text"]},
                margin={"l": 10, "r": 10, "t": 45, "b": 10},
            )
            fig_adop.update_xaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
            fig_adop.update_yaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
            st.plotly_chart(fig_adop, use_container_width=True)

        if "monto" in importance_map:
            fig_monto = px.bar(
                importance_map["monto"].sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                title="Top variables | Modelo de monto",
                color_discrete_sequence=[PALETTE["priority"]],
                template="plotly_white",
            )
            fig_monto.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": PALETTE["text"]},
                margin={"l": 10, "r": 10, "t": 45, "b": 10},
            )
            fig_monto.update_xaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
            fig_monto.update_yaxes(title_font={"color": PALETTE["text"]}, tickfont={"color": PALETTE["text"]})
            st.plotly_chart(fig_monto, use_container_width=True)

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

    if rows_feature_store:
        st.caption(
            f"Feature store detectado: {rows_feature_store:,} filas x {cols_feature_store:,} columnas."
        )


def main() -> None:
    st.set_page_config(
        page_title="CREAN | App de Inversiones",
        page_icon="📊",
        layout="wide",
    )
    inject_css()

    st.title("App de Inversiones CREAN")
    st.caption("Plataforma de priorizacion comercial y diagnostico MLOps")

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
    else:
        render_module_technical(filtered_df)


if __name__ == "__main__":
    main()