"""
Hospital Infection & AMR Intelligence Dashboard
----------------------------------------------------
Run with: streamlit run dashboard.py
Design: pastel KPI cards + purple "Metrics Overview" lollipop chart card,
quick-actions list, and a colorful case list -- with a light/dark toggle.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import io
import base64

DATA_PATH = "data/armd_with_network_features.parquet"
RANDOM_SEED = 42
NETWORK_COLS = ["network_degree", "network_degree_centrality", "network_cluster_size"]

st.set_page_config(
    page_title="Hospital Infection Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME -- dark mode only
# ============================================================
IS_DARK = True

PAGE_BG = "#0B0B10"
CARD_BG = "#15151C"
TEXT_MAIN = "#F1F1F4"
TEXT_MUTED = "#9AA0AE"
BORDER = "#2A2A33"
BLUE_CARD, BLUE_ICON, BLUE_ACC = "#16233A", "#22344F", "#5AA9F9"
OLIVE_CARD, OLIVE_ICON, OLIVE_ACC = "#26260F", "#39390F", "#D8CD5A"
PINK_CARD, PINK_ICON, PINK_ACC = "#2A1620", "#3A2029", "#F28FA3"
MINT_CARD, MINT_ICON, MINT_ACC = "#122420", "#1C332C", "#5FD9B0"
PURPLE_1, PURPLE_2 = "#4338CA", "#7C3AED"
BAR_COLOR, CIRCLE_COLOR = "#38BDF8", "#22D3EE"
RISK, SAFE, WARN = "#F87171", "#34D399", "#FBBF24"

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>
    .main {{ background-color: {PAGE_BG}; }}
    h1, h2, h3, h4 {{ color: {TEXT_MAIN}; font-family: 'Segoe UI', sans-serif; font-weight: 700; }}
    p, div, span, label {{ font-family: 'Segoe UI', sans-serif; }}

    .kpi-card {{
        border-radius: 18px; padding: 18px 20px; height: 165px;
        display: flex; flex-direction: column; justify-content: space-between;
        transition: transform 0.15s ease;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-icon {{
        width: 38px; height: 38px; border-radius: 12px; display: flex;
        align-items: center; justify-content: center; font-size: 1.1rem; margin-bottom: 8px;
    }}
    .kpi-title {{ font-weight: 700; font-size: 1.02rem; color: {TEXT_MAIN}; }}
    .kpi-subtitle {{ font-size: 0.82rem; color: {TEXT_MUTED}; margin-top: 2px; }}
    .kpi-bottom {{ display: flex; align-items: center; justify-content: space-between; margin-top: 8px; }}
    .kpi-number {{ font-size: 1.7rem; font-weight: 800; color: {TEXT_MAIN}; }}
    .kpi-pill {{
        display: flex; align-items: center; gap: 4px; font-size: 0.78rem; font-weight: 700;
        padding: 4px 9px; border-radius: 20px; background: rgba(0,0,0,0.06); color: {TEXT_MAIN};
    }}

    .purple-card {{
        background: linear-gradient(135deg, {PURPLE_1} 0%, {PURPLE_2} 100%);
        border-radius: 18px; padding: 22px 24px; height: 340px;
    }}
    .purple-card h3 {{ color: white !important; margin: 0 0 10px 0; }}

    .actions-card {{
        background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 18px;
        padding: 16px 18px; margin-bottom: 10px;
    }}
    .action-row {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 4px; border-bottom: 1px solid {BORDER};
    }}
    .action-row:last-child {{ border-bottom: none; }}
    .action-left {{ display: flex; align-items: center; gap: 12px; }}
    .action-icon {{
        width: 36px; height: 36px; border-radius: 10px; display: flex;
        align-items: center; justify-content: center; font-size: 1rem;
    }}
    .action-label {{ font-weight: 600; color: {TEXT_MAIN}; font-size: 0.95rem; }}

    .case-list-card {{
        background: {CARD_BG}; border-radius: 18px; padding: 4px;
        background: linear-gradient(135deg, {PINK_ACC}22, {PURPLE_1}22, {WARN}22);
        border: 1px solid {BORDER};
    }}
    .case-row {{
        background: {CARD_BG}; border-radius: 14px; padding: 12px 16px; margin: 6px;
        display: flex; align-items: center; justify-content: space-between;
    }}
    .case-left {{ display: flex; align-items: center; gap: 12px; }}
    .case-avatar {{
        width: 38px; height: 38px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-size: 1rem; color: white;
        background: linear-gradient(135deg, {PURPLE_1}, {PURPLE_2});
    }}
    .case-name {{ font-weight: 700; color: {TEXT_MAIN}; font-size: 0.92rem; }}
    .case-sub {{ font-size: 0.78rem; color: {TEXT_MUTED}; }}
    .status-pill {{
        font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 14px;
    }}

    div[data-testid="stMetric"] {{
        background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 16px;
        padding: 16px 18px 12px 18px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; }}
    div[data-testid="stMetricValue"] {{ color: {TEXT_MAIN}; font-weight: 800; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: 10px 16px; font-weight: 600; color: {TEXT_MUTED}; }}
    .stTabs [aria-selected="true"] {{ background: {PURPLE_1}; color: white !important; border-radius: 10px; }}

    div[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; border: 1px solid {BORDER}; }}
    section[data-testid="stSidebar"] {{ background: {CARD_BG}; border-right: 1px solid {BORDER}; }}
    section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN} !important; }}
    .disclaimer-box {{
        background: {WARN}18; border: 1px solid {WARN}55; padding: 12px 16px;
        border-radius: 12px; font-size: 0.82rem; color: {TEXT_MAIN}; margin-top: 8px;
    }}

    /* Fix low-contrast text in the main content area -- Streamlit's own
       base theme can otherwise leave widget labels/captions too light or
       too dark depending on the browser/system theme, independent of our
       app-level light/dark toggle. Force everything to our palette. */
    .block-container, .stMarkdown, .stMarkdown p {{ color: {TEXT_MAIN}; }}
    label, .stSelectbox label, .stSlider label, .stRadio label,
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] {{
        color: {TEXT_MAIN} !important;
    }}
    [data-testid="stCaptionContainer"], .stCaption {{ color: {TEXT_MUTED} !important; }}
    div[data-baseweb="select"] > div {{
        background: {CARD_BG} !important; border-color: {BORDER} !important; color: {TEXT_MAIN} !important;
    }}
    div[data-baseweb="select"] span {{ color: {TEXT_MAIN} !important; }}
    .stRadio div[role="radiogroup"] label span {{ color: {TEXT_MAIN} !important; }}

    .stButton > button {{
        background: {CARD_BG}; color: {TEXT_MAIN}; border: 1px solid {BORDER};
        border-radius: 10px; font-weight: 600;
    }}
    .stButton > button:hover {{ border-color: {PURPLE_1}; color: {PURPLE_1}; }}
    .stButton > button p {{ color: inherit !important; }}
</style>
""", unsafe_allow_html=True)


def new_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAGE_BG)
    ax.set_facecolor(PAGE_BG)
    return fig, ax


def style_ax(ax, title, xlabel, ylabel, rotate_x=0):
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_MAIN, pad=14, loc="left")
    ax.set_xlabel(xlabel, fontsize=9.5, color=TEXT_MUTED, fontweight="600")
    ax.set_ylabel(ylabel, fontsize=9.5, color=TEXT_MUTED, fontweight="600")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    if rotate_x:
        plt.setp(ax.get_xticklabels(), rotation=rotate_x, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25, color=TEXT_MUTED, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def show_chart(fig):
    st.pyplot(fig)


def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 PNG string, so it can be
    embedded directly inside a custom HTML block (needed because content
    rendered between two separate st.markdown() calls does NOT nest inside
    the div they define -- each st.markdown() is its own isolated block)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=140)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{encoded}"


def kpi_card(icon, icon_bg, card_bg, title, subtitle, number, pill_text, pill_color):
    st.markdown(f"""
    <div class="kpi-card" style="background:{card_bg};">
        <div>
            <div class="kpi-icon" style="background:{icon_bg};">{icon}</div>
            <div class="kpi-title">{title}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        <div class="kpi-bottom">
            <div class="kpi-number">{number}</div>
            <div class="kpi-pill" style="color:{pill_color};">{pill_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def lollipop_chart(labels, values, bar_color, circle_color):
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    x = np.arange(len(labels))
    ax.vlines(x, 0, values, color=bar_color, linewidth=7, alpha=0.85)
    ax.scatter(x, values, s=750, color=circle_color, zorder=3, edgecolors="white", linewidths=1.5)
    for xi, v in zip(x, values):
        ax.text(xi, v, f"{int(v)}", ha="center", va="center", fontsize=9,
                fontweight="800", color="#0B1220", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9.5, fontweight="600")
    ax.tick_params(axis="y", colors="white", labelsize=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.grid(axis="y", color="white", alpha=0.12, linestyle="-")
    ax.set_ylim(0, max(values) * 1.25)
    return fig


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


def prepare_features(df):
    target_col = "outbreak_flag"
    y = df[target_col]
    id_and_meta_cols = [
        "anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded",
        "order_time_jittered_utc", "anon_id_ward", "pat_enc_csn_id_coded_ward",
        "order_time_jittered_utc_ward", "ward_type", "organism",
        "culture_description", target_col
    ]
    id_and_meta_cols = [c for c in id_and_meta_cols if c in df.columns]
    X = df.drop(columns=id_and_meta_cols)
    X = X.select_dtypes(include=[np.number])
    id_like_cols = [c for c in X.columns if
                    "anon_id" in c.lower() or "pat_enc_csn_id" in c.lower() or "order_proc_id" in c.lower()]
    X = X.drop(columns=id_like_cols)
    return X, y


@st.cache_resource
def train_model(df):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(n_estimators=200, scale_pos_weight=pos_weight,
                           random_state=RANDOM_SEED, eval_metric="logloss")
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_test, y_test, X.columns.tolist()


@st.cache_resource
def train_monotonic_model(df):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    monotone_map = {c: 1 for c in NETWORK_COLS}
    constraints = tuple(monotone_map.get(col, 0) for col in X.columns)
    model = XGBClassifier(n_estimators=200, scale_pos_weight=pos_weight,
                           random_state=RANDOM_SEED, eval_metric="logloss",
                           monotone_constraints=constraints)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X.columns.tolist()


def isolate_case(case, network_cols):
    case_isolated = case.copy()
    if "network_degree" in case_isolated.columns:
        case_isolated["network_degree"] = 0
    if "network_degree_centrality" in case_isolated.columns:
        case_isolated["network_degree_centrality"] = 0
    if "network_cluster_size" in case_isolated.columns:
        case_isolated["network_cluster_size"] = 1
    return case_isolated


@st.cache_data
def compute_amr_trends(df, top_n=5, forecast_window=3):
    time_col = "order_time_jittered_utc"
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["period"] = df[time_col].dt.to_period("M")

    amr_cols = [c for c in df.columns if c.startswith("AMR__")]
    if len(amr_cols) == 0:
        return {}, []

    occurrence_counts = (df[amr_cols] != -1).sum().sort_values(ascending=False)
    top_pairs = occurrence_counts.head(top_n)

    results = {}
    for col, count in top_pairs.items():
        pair_name = col.replace("AMR__", "")
        period_group = df.groupby("period", observed=True).apply(
            lambda g: (g[col] != -1).sum() / len(g) if len(g) > 0 else 0,
            include_groups=False
        ).sort_index()
        if len(period_group) < 2:
            continue
        forecast_next = period_group.tail(forecast_window).mean()
        early_avg = period_group.head(len(period_group) // 3).mean()
        late_avg = period_group.tail(len(period_group) // 3).mean()
        trend_direction = "Rising" if late_avg > early_avg * 1.2 else (
            "Falling" if late_avg < early_avg * 0.8 else "Relatively Flat"
        )
        results[pair_name] = {
            "series": period_group, "count": count, "forecast": forecast_next,
            "early_avg": early_avg, "late_avg": late_avg, "trend": trend_direction,
        }
    return results, list(top_pairs.index)


@st.cache_resource
def build_graph(df):
    time_col = "order_time_jittered_utc"
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    ward_flag_cols = ["hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU"]
    ward_flag_cols = [c for c in ward_flag_cols if c in df.columns]

    def get_ward(row):
        for c in ward_flag_cols:
            if row[c] == 1:
                return c.replace("hosp_ward_", "")
        return "UNKNOWN"

    df["ward_type"] = df.apply(get_ward, axis=1)
    OVERLAP_WINDOW_DAYS = 1
    MEANINGFUL_WARDS = ["IP", "ICU"]
    G = nx.Graph()
    G.add_nodes_from(df["order_proc_id_coded"])
    for ward, group in df.groupby("ward_type"):
        if ward not in MEANINGFUL_WARDS:
            continue
        group = group.sort_values(time_col).reset_index(drop=True)
        times = group[time_col].values
        ids = group["order_proc_id_coded"].values
        for i in range(len(times)):
            window_end = times[i] + np.timedelta64(OVERLAP_WINDOW_DAYS, "D")
            connected_mask = (times > times[i]) & (times <= window_end)
            for other_id in ids[connected_mask]:
                G.add_edge(ids[i], other_id)
    components = list(nx.connected_components(G))
    return G, components


# ============================================================
# LOAD DATA + MODELS
# ============================================================
df = load_data()
model, explainer, X_test, y_test, feature_cols = train_model(df)
y_proba = model.predict_proba(X_test)[:, 1]

mono_model, mono_X_test, mono_y_test, mono_feature_cols = train_monotonic_model(df)
mono_y_proba = mono_model.predict_proba(mono_X_test)[:, 1]

G, components = build_graph(df)
components_sorted = sorted(components, key=len, reverse=True)

amr_trends, amr_pair_cols = compute_amr_trends(df)

# --- Real, non-fabricated trend: compare first half vs second half of the sample ---
time_col = "order_time_jittered_utc"
df_sorted = df.copy()
df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], errors="coerce")
df_sorted = df_sorted.sort_values(time_col)
half = len(df_sorted) // 2
first_half, second_half = df_sorted.iloc[:half], df_sorted.iloc[half:]

def pct_change(first, second):
    if first == 0:
        return 0
    return (second - first) / first * 100

positive_trend = pct_change((first_half["was_positive"] == 1).sum(), (second_half["was_positive"] == 1).sum())
outbreak_trend = pct_change((first_half["outbreak_flag"] == 1).sum(), (second_half["outbreak_flag"] == 1).sum())

# ============================================================
# SIDEBAR CONTENT
# ============================================================
with st.sidebar:
    st.markdown(f"<h2>🏥 InfectionIQ</h2>", unsafe_allow_html=True)
    st.caption("AI-based hospital infection & AMR decision-support")
    st.divider()
    st.markdown(f"""
    <div style="line-height:2.1;">
    🦠&nbsp;&nbsp;Infection Monitor<br>
    📈&nbsp;&nbsp;Risk & Forecast<br>
    🧠&nbsp;&nbsp;Explainability<br>
    🧮&nbsp;&nbsp;Intervention Simulator<br>
    🚨&nbsp;&nbsp;Resource Optimizer<br>
    🕸️&nbsp;&nbsp;Transmission Network<br>
    🧬&nbsp;&nbsp;AMR Trends
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**Dataset**")
    st.write(f"Source: ARMD (Stanford)")
    st.write(f"Sample: {len(df):,} culture orders")
    st.write(f"Features: {len(feature_cols)}")
    st.divider()
    st.markdown(
        "<div class='disclaimer-box'>⚠️ Model-based estimates for research/"
        "decision-support only. Not a clinical diagnostic tool.</div>",
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(f"<h1 style='margin-bottom:0;'>🏥 Hospital Infection & AMR Intelligence</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{TEXT_MUTED}; margin-top:0;'>Surveillance, forecasting, and decision support for hospital-acquired infections and antimicrobial resistance.</p>", unsafe_allow_html=True)
st.write("")

# ============================================================
# HERO LAYOUT -- quick actions + KPI cards (left), purple metrics card (right)
# ============================================================
left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.markdown('<div class="actions-card">', unsafe_allow_html=True)
    actions = [
        ("🔍", BLUE_ICON, "Run New Risk Scan", "scan", "📈 Scroll down to the 'Risk & Forecast' tab to see current predictions."),
        ("🧪", MINT_ICON, "View AMR Trend Report", "amr", "🧬 Scroll down to the 'AMR Trends' tab for resistance trend charts."),
        ("🕸️", PINK_ICON, "Inspect Network Clusters", "net", "🕸️ Scroll down to the 'Transmission Network' tab to view clusters."),
    ]
    for icon, icon_bg, label, key, hint in actions:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"""
            <div class="action-row" style="border-bottom:none;">
                <div class="action-left">
                    <div class="action-icon" style="background:{icon_bg};">{icon}</div>
                    <div class="action-label">{label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            if st.button("Go", key=f"btn_{key}"):
                st.toast(hint, icon="👉")
    st.markdown('</div>', unsafe_allow_html=True)

    kc1, kc2 = st.columns(2)
    with kc1:
        kpi_card("🧫", BLUE_ICON, BLUE_CARD, "Positive Cultures", "Confirmed infections",
                 f"{(df['was_positive']==1).sum():,}",
                 f"{'↑' if positive_trend >= 0 else '↓'} {abs(positive_trend):.0f}%", BLUE_ACC)
    with kc2:
        kpi_card("🚨", PINK_ICON, PINK_CARD, "Outbreak Signals", "Flagged clusters",
                 f"{(df['outbreak_flag']==1).sum():,}",
                 f"{'↑' if outbreak_trend >= 0 else '↓'} {abs(outbreak_trend):.0f}%", PINK_ACC)

    kc3, kc4 = st.columns(2)
    with kc3:
        kpi_card("🧬", MINT_ICON, MINT_CARD, "AMR Pairs Tracked", "Organism-antibiotic",
                 f"{len(amr_trends)}", "Live", MINT_ACC)
    with kc4:
        kpi_card("👥", OLIVE_ICON, OLIVE_CARD, "Total Culture Orders", "Sample size",
                 f"{len(df):,}", "10K target", OLIVE_ACC)

with right_col:
    monthly = df_sorted.groupby(df_sorted[time_col].dt.to_period("M")).size().tail(7)
    labels = [p.strftime("%b") for p in monthly.index]
    fig = lollipop_chart(labels, monthly.values, BAR_COLOR, CIRCLE_COLOR)
    chart_uri = fig_to_base64(fig)

    st.markdown(f"""
    <div class="purple-card">
        <h3>📊 Metrics Overview</h3>
        <p style="color:rgba(255,255,255,0.75); font-size:0.85rem; margin-top:-8px;">
            Culture volume, most recent 7 months in sample
        </p>
        <img src="{chart_uri}" style="width:100%; margin-top:12px;">
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --- Top risk case list, styled like the reference's colorful-bordered list ---
st.markdown('<div class="case-list-card">', unsafe_allow_html=True)
top3_idx = np.argsort(y_proba)[-3:][::-1]
status_map = [("High Risk", RISK), ("Medium Risk", WARN), ("Watch", SAFE)]
for rank, idx in enumerate(top3_idx):
    label, color = status_map[min(rank, 2)]
    st.markdown(f"""
    <div class="case-row">
        <div class="case-left">
            <div class="case-avatar">🧫</div>
            <div>
                <div class="case-name">Test Case #{idx}</div>
                <div class="case-sub">Predicted risk: {y_proba[idx]:.1%} · Actual: {"Outbreak" if y_test.iloc[idx]==1 else "No Outbreak"}</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <span class="status-pill" style="background:{color}22; color:{color};">{label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.divider()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🦠 Infection Monitor", "📈 Risk & Forecast", "🧠 Explainability",
    "🧮 Intervention Simulator", "🚨 Resource Optimizer", "🕸️ Transmission Network",
    "🧬 AMR Trends"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top Organisms (Positive Cultures)")
        if "organism" in df.columns:
            top_organisms = df[df["was_positive"] == 1]["organism"].value_counts().head(10)
            fig, ax = new_figure((6, 4.2))
            ax.barh(top_organisms.index[::-1], top_organisms.values[::-1], color=BLUE_ACC)
            style_ax(ax, "Top 10 Organisms Identified", "Number of Positive Cultures", "Organism")
            show_chart(fig)
    with c2:
        st.markdown("#### Ward Distribution")
        ward_cols = ["hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU"]
        ward_counts = {c.replace("hosp_ward_", ""): df[c].sum() for c in ward_cols if c in df.columns}
        fig, ax = new_figure((6, 4.2))
        ax.bar(ward_counts.keys(), ward_counts.values(), color=BLUE_ACC, width=0.55)
        style_ax(ax, "Culture Orders by Ward Setting", "Ward Setting", "Number of Culture Orders")
        show_chart(fig)

with tab2:
    st.markdown("#### Model Risk Predictions (Test Set)")
    risk_df = pd.DataFrame({
        "Predicted Risk": y_proba, "Actual Outbreak": y_test.values
    }).sort_values("Predicted Risk", ascending=False).reset_index(drop=True)
    display_risk = risk_df.head(20).copy()
    display_risk["Predicted Risk"] = display_risk["Predicted Risk"].apply(lambda x: f"{x:.2%}")
    display_risk["Actual Outbreak"] = display_risk["Actual Outbreak"].map({1: "Yes", 0: "No"})
    st.dataframe(display_risk, use_container_width=True, hide_index=True)

    st.markdown("#### Risk Score Distribution")
    fig, ax = new_figure((10, 4))
    ax.hist(y_proba, bins=20, color=BLUE_ACC, alpha=0.9)
    style_ax(ax, "Distribution of Predicted Outbreak Risk (Test Set)", "Predicted Risk Score", "Number of Culture Orders")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    show_chart(fig)

with tab3:
    st.markdown("#### Why is this case high-risk?")
    st.caption("SHAP values show how much each feature pushed this specific prediction up or down.")
    selected_idx = st.selectbox(
        "Select a test case (ranked by predicted risk, highest first):",
        options=range(min(20, len(X_test))),
        format_func=lambda i: f"Case {i+1} — Predicted risk: {y_proba[np.argsort(y_proba)[::-1][i]]:.2%}",
        key="explain_select"
    )
    real_idx = np.argsort(y_proba)[::-1][selected_idx]
    shap_values = explainer.shap_values(X_test.iloc[[real_idx]])
    case_shap = pd.Series(shap_values[0], index=feature_cols).sort_values(key=abs, ascending=False).head(10)

    col1, col2 = st.columns(2)
    col1.metric("Predicted Risk", f"{y_proba[real_idx]:.2%}")
    col2.metric("Actual Outcome", "Outbreak" if y_test.iloc[real_idx] == 1 else "No Outbreak")

    fig, ax = new_figure((9, 5))
    colors = [RISK if v > 0 else SAFE for v in case_shap.values]
    ax.barh(case_shap.index[::-1], case_shap.values[::-1], color=colors[::-1])
    style_ax(ax, "Top 10 Contributing Factors (SHAP)", "SHAP Value (Impact on Predicted Risk)", "Feature")
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.8)
    show_chart(fig)
    st.caption("🔴 Red = increases predicted risk    🟢 Green = decreases predicted risk")

with tab4:
    st.markdown("#### What-If Intervention Simulator")
    st.caption("Uses a monotonic-constrained model so reduced exposure never increases predicted risk.")
    sim_idx = st.selectbox(
        "Select a case to simulate interventions for:",
        options=range(min(20, len(mono_X_test))),
        format_func=lambda i: f"Case {i+1} — Baseline risk: {mono_y_proba[np.argsort(mono_y_proba)[::-1][i]]:.2%}",
        key="sim_select"
    )
    real_sim_idx = np.argsort(mono_y_proba)[::-1][sim_idx]
    case = mono_X_test.iloc[[real_sim_idx]].copy()
    baseline_risk = mono_y_proba[real_sim_idx]
    case_isolated = isolate_case(case, NETWORK_COLS)
    isolated_risk = mono_model.predict_proba(case_isolated)[0, 1]
    case_reduced = case.copy()
    if "network_degree" in case_reduced.columns:
        case_reduced["network_degree"] = np.floor(case_reduced["network_degree"] / 2)
    if "network_cluster_size" in case_reduced.columns:
        case_reduced["network_cluster_size"] = np.maximum(1, np.floor(case_reduced["network_cluster_size"] / 2))
    reduced_risk = mono_model.predict_proba(case_reduced)[0, 1]

    scenario_names = ["No Intervention", "Reduced Exposure\n(Halved)", "Full Isolation"]
    scenario_values = [baseline_risk, reduced_risk, isolated_risk]
    fig, ax = new_figure((8, 4.5))
    bars = ax.bar(scenario_names, scenario_values, color=[RISK, WARN, SAFE], width=0.5)
    for bar, val in zip(bars, scenario_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.1%}", ha="center", fontweight="bold", color=TEXT_MAIN)
    style_ax(ax, "Predicted Risk Under Different Intervention Scenarios", "Scenario", "Predicted Risk")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    show_chart(fig)

with tab5:
    st.markdown("#### Resource-Constrained Optimizer")
    st.caption("Given a limited number of isolation beds, ranks candidates by estimated risk reduction.")
    c1, c2 = st.columns(2)
    n_beds = c1.slider("Isolation beds available", min_value=1, max_value=20, value=5)
    n_candidates = c2.slider("Number of high-risk candidates to consider", min_value=5, max_value=50, value=20)

    top_candidates_positions = np.argsort(mono_y_proba)[-n_candidates:][::-1]
    candidate_results = []
    for pos in top_candidates_positions:
        case = mono_X_test.iloc[[pos]].copy()
        baseline_risk = mono_y_proba[pos]
        case_isolated = isolate_case(case, NETWORK_COLS)
        isolated_risk = mono_model.predict_proba(case_isolated)[0, 1]
        candidate_results.append({
            "Test Case #": pos, "Baseline Risk": baseline_risk,
            "Risk if Isolated": isolated_risk, "Risk Reduction": baseline_risk - isolated_risk,
        })
    candidates_df = pd.DataFrame(candidate_results).sort_values("Risk Reduction", ascending=False).reset_index(drop=True)
    candidates_df["Allocated"] = ["✅ Isolate" if i < n_beds else "—" for i in range(len(candidates_df))]

    allocated = candidates_df[candidates_df["Allocated"] == "✅ Isolate"]
    avg_reduction = allocated["Risk Reduction"].mean()
    total_reduction_allocated = allocated["Risk Reduction"].sum()
    total_reduction_all = candidates_df["Risk Reduction"].sum()
    efficiency = (total_reduction_allocated / total_reduction_all * 100) if total_reduction_all > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg. Risk Reduction / Bed", f"{avg_reduction:.1%}")
    m2.metric("Beds Used", f"{n_beds} / {n_candidates}")
    m3.metric("Efficiency Captured", f"{efficiency:.1f}%")

    fig, ax = new_figure((10, 5))
    bar_colors = [SAFE if a == "✅ Isolate" else BORDER for a in candidates_df["Allocated"]]
    ax.bar(range(len(candidates_df)), candidates_df["Risk Reduction"], color=bar_colors)
    style_ax(ax, "Risk Reduction by Candidate (Green = Allocated a Bed)", "Candidate (Ranked by Risk Reduction)", "Risk Reduction")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    show_chart(fig)

    display_candidates = candidates_df.copy()
    for col in ["Baseline Risk", "Risk if Isolated", "Risk Reduction"]:
        display_candidates[col] = display_candidates[col].apply(lambda x: f"{x:.2%}")
    st.dataframe(display_candidates, use_container_width=True, hide_index=True)

with tab6:
    st.markdown("#### Patient-Contact Network")
    st.caption("Connections represent potential exposure: same inpatient/ICU ward within 1 day.")
    non_trivial = [c for c in components_sorted if len(c) > 1]
    isolated_count = sum(1 for c in components if len(c) == 1)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Nodes", f"{G.number_of_nodes():,}")
    m2.metric("Total Connections", f"{G.number_of_edges():,}")
    m3.metric("Isolated (No Connections)", f"{isolated_count:,}")

    st.markdown("#### Largest Potential Clusters")
    top10 = non_trivial[:10]
    fig, ax = new_figure((10, 4))
    ax.bar(range(1, len(top10) + 1), [len(c) for c in top10], color=RISK)
    style_ax(ax, "Top 10 Largest Potential Exposure Clusters", "Cluster Rank", "Cluster Size")
    ax.set_xticks(range(1, len(top10) + 1))
    show_chart(fig)

    st.markdown("#### Visualize a Cluster")
    if len(non_trivial) > 0:
        cluster_choice = st.selectbox(
            "Select a cluster to visualize:",
            options=range(min(5, len(non_trivial))),
            format_func=lambda i: f"Cluster {i+1} — {len(non_trivial[i])} connected culture orders"
        )
        chosen_nodes = non_trivial[cluster_choice]
        subgraph = G.subgraph(chosen_nodes)
        fig, ax = new_figure((8, 6))
        pos = nx.spring_layout(subgraph, seed=RANDOM_SEED)
        nx.draw(subgraph, pos, ax=ax, with_labels=False, node_color=BLUE_ACC,
                node_size=260, edge_color=TEXT_MUTED, alpha=0.9, linewidths=1.5, edgecolors=PAGE_BG)
        ax.set_title(f"Cluster {cluster_choice + 1}: {len(chosen_nodes)} Connected Culture Orders",
                     fontsize=13, fontweight="bold", color=TEXT_MAIN, pad=12, loc="left")
        show_chart(fig)
    else:
        st.info("No multi-node clusters found in this sample.")

with tab7:
    st.markdown("#### Antimicrobial Resistance Trends")
    st.caption("Resistance rate over time for the most frequent organism-antibiotic pairs.")
    if len(amr_trends) == 0:
        st.warning("No AMR__ columns with enough occurrences were found in this sample.")
    else:
        pair_choice = st.selectbox("Select an organism-antibiotic pair:", options=list(amr_trends.keys()))
        data = amr_trends[pair_choice]
        series = data["series"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Occurrences", f"{data['count']:,}")
        m2.metric("Trend", data["trend"])
        m3.metric("Forecast (Next Period)", f"{data['forecast']:.2%}")

        fig, ax = new_figure((11, 4.5))
        x_labels = [str(p) for p in series.index]
        ax.plot(x_labels, series.values, color=BLUE_ACC, linewidth=2, marker="o", markersize=4,
                markerfacecolor=CIRCLE_COLOR, markeredgecolor=PAGE_BG, markeredgewidth=1)
        ax.fill_between(range(len(x_labels)), series.values, color=BLUE_ACC, alpha=0.1)
        ax.axhline(data["forecast"], color=RISK, linestyle="--", linewidth=1.6,
                   label=f"Forecast next period ({data['forecast']:.1%})")
        style_ax(ax, f"Resistance Rate Over Time — {pair_choice}", "Time Period (Monthly)", "Resistance Rate")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        n_ticks = 12
        step = max(1, len(x_labels) // n_ticks)
        ax.set_xticks(range(0, len(x_labels), step))
        ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45, ha="right")
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        show_chart(fig)

        st.caption(f"Time span: {series.index.min()} to {series.index.max()} ({len(series)} months).")

st.divider()
st.markdown(f"<p style='text-align:center; color:{TEXT_MUTED}; font-size:0.82rem;'>Hospital Infection & AMR Intelligence Dashboard · Model-based decision support · Not a clinical diagnostic tool</p>", unsafe_allow_html=True)