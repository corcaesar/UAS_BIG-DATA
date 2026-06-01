import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Big Data Sentiment Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_ACCURACY = 52.33
DEFAULT_TOTAL_DATA = 13435
DEFAULT_ENV_DATA = 4316
PREDICTION_MAP = {
    "0": "positive",
    "0.0": "positive",
    0: "positive",
    0.0: "positive",
    "1": "negative",
    "1.0": "negative",
    1: "negative",
    1.0: "negative",
    "2": "neutral",
    "2.0": "neutral",
    2: "neutral",
    2.0: "neutral",
}
SENTIMENT_ORDER = ["positive", "negative", "neutral"]


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names without forcing case, so original names remain readable."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in lower_map:
            return lower_map[cand.lower().strip()]
    return None


def prepare_sentiment_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Return dataframe, sentiment column name, and basis description."""
    df = df.copy()

    predicted_col = get_col(df, ["predicted_sentiment", "predicted label", "prediction_label"])
    if predicted_col:
        df["sentiment_dashboard"] = df[predicted_col].astype(str).str.lower().str.strip()
        return df, "sentiment_dashboard", f"prediksi model ({predicted_col})"

    prediction_col = get_col(df, ["prediction", "predicted", "label_prediction"])
    if prediction_col:
        df["sentiment_dashboard"] = df[prediction_col].map(PREDICTION_MAP)
        df["sentiment_dashboard"] = df["sentiment_dashboard"].fillna(df[prediction_col].astype(str).str.lower().str.strip())
        return df, "sentiment_dashboard", f"kode prediksi model ({prediction_col})"

    final_label_col = get_col(df, ["final label", "final_label", "sentiment", "label"])
    if final_label_col:
        df["sentiment_dashboard"] = df[final_label_col].astype(str).str.lower().str.strip()
        return df, "sentiment_dashboard", f"label dataset ({final_label_col})"

    return df, "", "tidak ditemukan"


def build_sample_data() -> pd.DataFrame:
    rows = []
    summary = [
        ("positive", 495, 60.810811),
        ("negative", 306, 37.592138),
        ("neutral", 13, 1.597052),
    ]
    keyword_pool = {
        "positive": ["electric car", "recycle", "climate change", "forest fires"],
        "negative": ["climate change", "deforestation", "forest fires", "recycle"],
        "neutral": ["climate change", "wildlife reserve", "recycle"],
    }
    idx = 1
    for sentiment, count, _ in summary:
        words = keyword_pool[sentiment]
        for i in range(count):
            rows.append(
                {
                    "id": f"sample-{idx}",
                    "keyword": words[i % len(words)],
                    "subdomain": "environmental management",
                    "prediction": {"positive": 0.0, "negative": 1.0, "neutral": 2.0}[sentiment],
                    "predicted_sentiment": sentiment,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def sentiment_summary(df: pd.DataFrame, sentiment_col: str) -> pd.DataFrame:
    summary = (
        df[sentiment_col]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    total = summary["count"].sum()
    summary["percentage"] = (summary["count"] / total * 100).round(2) if total else 0
    summary["sentiment"] = pd.Categorical(summary["sentiment"], categories=SENTIMENT_ORDER, ordered=True)
    summary = summary.sort_values(["sentiment", "count"], ascending=[True, False])
    summary["sentiment"] = summary["sentiment"].astype(str)
    return summary


def format_pct(value: float) -> str:
    return f"{value:.2f}%".replace(".", ",")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🌱 Big Data Sentiment")
st.sidebar.caption("Dashboard analisis sentimen isu lingkungan berbasis dataset biodiversity.")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV hasil Colab",
    type=["csv"],
    help="Bisa upload sentiment_results.csv dari Colab atau biodiversity_for_modelling.csv.",
)

local_candidates = [
    Path("data/sentiment_results.csv"),
    Path("data/biodiversity_for_modelling.csv"),
    Path("sentiment_results.csv"),
    Path("biodiversity_for_modelling.csv"),
]

source_label = "sample data dari hasil notebook"
if uploaded_file is not None:
    raw_df = load_csv_from_upload(uploaded_file)
    source_label = f"file upload: {uploaded_file.name}"
else:
    existing = next((p for p in local_candidates if p.exists()), None)
    if existing:
        raw_df = load_csv_from_path(str(existing))
        source_label = f"file lokal: {existing}"
    else:
        raw_df = build_sample_data()

raw_df = normalize_columns(raw_df)
prepared_df, sentiment_col, basis = prepare_sentiment_column(raw_df)

accuracy = st.sidebar.number_input(
    "Accuracy model (%)",
    min_value=0.0,
    max_value=100.0,
    value=DEFAULT_ACCURACY,
    step=0.01,
)

subdomain_col = get_col(prepared_df, ["subdomain"])
keyword_col = get_col(prepared_df, ["keyword", "keywords"])

filtered_df = prepared_df.copy()
if subdomain_col:
    subdomains = sorted(filtered_df[subdomain_col].dropna().astype(str).unique().tolist())
    default_subdomains = [s for s in subdomains if s.lower() == "environmental management"] or subdomains
    selected_subdomains = st.sidebar.multiselect(
        "Filter subdomain",
        options=subdomains,
        default=default_subdomains,
    )
    if selected_subdomains:
        filtered_df = filtered_df[filtered_df[subdomain_col].astype(str).isin(selected_subdomains)]

if keyword_col:
    keywords = sorted(filtered_df[keyword_col].dropna().astype(str).unique().tolist())
    selected_keywords = st.sidebar.multiselect(
        "Filter keyword",
        options=keywords,
        default=[],
    )
    if selected_keywords:
        filtered_df = filtered_df[filtered_df[keyword_col].astype(str).isin(selected_keywords)]

# -----------------------------
# Main layout
# -----------------------------
st.title("Analisis Sentimen Publik terhadap Isu Lingkungan")
st.markdown(
    "Dashboard ini menampilkan ringkasan hasil klasifikasi sentimen pada subdomain **environmental management** "
    "menggunakan pipeline Big Data Analytics berbasis PySpark dan Logistic Regression."
)

if not sentiment_col:
    st.error("Kolom sentimen/prediksi tidak ditemukan. Pastikan CSV memiliki kolom 'prediction', 'predicted_sentiment', atau 'final label'.")
    st.stop()

summary_df = sentiment_summary(filtered_df, sentiment_col)
if summary_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter.")
    st.stop()

# KPI cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Data Aktif", f"{len(filtered_df):,}".replace(",", "."))
with col2:
    st.metric("Data Environmental Management", f"{DEFAULT_ENV_DATA:,}".replace(",", "."))
with col3:
    st.metric("Accuracy Model", format_pct(float(accuracy)))
with col4:
    top_row = summary_df.sort_values("count", ascending=False).iloc[0]
    st.metric("Sentimen Dominan", f"{top_row['sentiment'].title()} ({format_pct(top_row['percentage'])})")

st.caption(f"Sumber data: {source_label} | Basis sentimen: {basis}")

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Distribusi Sentimen")
    fig_bar = px.bar(
        summary_df,
        x="sentiment",
        y="percentage",
        text=summary_df["percentage"].map(lambda x: f"{x:.2f}%"),
        labels={"sentiment": "Sentimen", "percentage": "Persentase (%)"},
        title="Persentase Sentimen",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(yaxis_range=[0, max(100, summary_df["percentage"].max() + 10)])
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Proporsi Sentimen")
    fig_pie = px.pie(
        summary_df,
        names="sentiment",
        values="count",
        hole=0.35,
        title="Proporsi Jumlah Data",
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Tabel Ringkasan Sentimen")
display_summary = summary_df.copy()
display_summary["percentage"] = display_summary["percentage"].map(format_pct)
st.dataframe(display_summary, use_container_width=True, hide_index=True)

if keyword_col:
    st.subheader("Keyword yang Paling Banyak Muncul")
    top_keywords = (
        filtered_df[keyword_col]
        .dropna()
        .astype(str)
        .value_counts()
        .head(10)
        .rename_axis("keyword")
        .reset_index(name="count")
    )
    fig_keyword = px.bar(
        top_keywords,
        x="count",
        y="keyword",
        orientation="h",
        text="count",
        labels={"count": "Jumlah", "keyword": "Keyword"},
        title="Top 10 Keyword",
    )
    fig_keyword.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_keyword, use_container_width=True)

    if sentiment_col:
        st.subheader("Perbandingan Keyword dan Sentimen")
        top_keyword_list = top_keywords["keyword"].tolist()
        keyword_sentiment = (
            filtered_df[filtered_df[keyword_col].astype(str).isin(top_keyword_list)]
            .groupby([keyword_col, sentiment_col])
            .size()
            .reset_index(name="count")
        )
        fig_stack = px.bar(
            keyword_sentiment,
            x=keyword_col,
            y="count",
            color=sentiment_col,
            barmode="stack",
            labels={keyword_col: "Keyword", "count": "Jumlah", sentiment_col: "Sentimen"},
            title="Distribusi Sentimen pada Keyword Teratas",
        )
        st.plotly_chart(fig_stack, use_container_width=True)

st.subheader("Insight Otomatis")
dominant = summary_df.sort_values("count", ascending=False).iloc[0]
neutral_row = summary_df[summary_df["sentiment"] == "neutral"]
neutral_pct = float(neutral_row["percentage"].iloc[0]) if not neutral_row.empty else 0.0
st.markdown(
    f"""
- Sentimen dominan adalah **{dominant['sentiment']}** dengan persentase **{format_pct(float(dominant['percentage']))}**.
- Nilai accuracy model saat ini adalah **{format_pct(float(accuracy))}**, sehingga model sudah dapat digunakan untuk demonstrasi pipeline, tetapi masih perlu optimasi fitur dan evaluasi lanjutan.
- Proporsi sentimen netral sebesar **{format_pct(neutral_pct)}**, sehingga hasil klasifikasi cenderung lebih banyak masuk ke kategori positif atau negatif.
"""
)

with st.expander("Preview Data"):
    st.dataframe(filtered_df.head(200), use_container_width=True)

with st.expander("Catatan penggunaan"):
    st.markdown(
        """
1. Untuk hasil paling sesuai laporan, upload CSV hasil ekspor dari Colab yang berisi kolom `prediction` atau `predicted_sentiment`.
2. Jika hanya memakai `biodiversity_for_modelling.csv`, dashboard akan membaca kolom `final label` sebagai label dataset.
3. Mapping default kode prediksi mengikuti output notebook: `0 = positive`, `1 = negative`, dan `2 = neutral`.
"""
    )
