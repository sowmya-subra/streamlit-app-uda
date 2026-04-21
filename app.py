"""
GuardRail — Streamlit App
6 tabs total. Live (parquet-only): Tab 02 Heatmap, Tab 04 Politeness Paradox, Tab 05 Topic Explorer.
Partial-live (parquet-backed fallback + auto-load hooks for Stage-7/8 artifacts):
    Tab 01 Attack Surface, Tab 03 Super-Attack, Tab 06 Risk Scorer.

Run:  streamlit run app.py
Deps: streamlit pandas numpy plotly pyarrow
"""

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GuardRail",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject a clean light theme with sharp typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp { background: #f8fafc; color: #0f172a; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }

    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    h1 { color: #ea580c; letter-spacing: -1px; }
    h3 { color: #334155; font-weight: 400; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; color: #ea580c; font-weight: 600; }
    .metric-label { font-size: 0.8rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

    .stub-banner {
        background: #fff7ed;
        border-left: 3px solid #f97316;
        padding: 0.8rem 1.2rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 1.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #9a3412;
    }

    div[data-testid="stTabs"] button {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #475569;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ea580c;
        border-bottom-color: #ea580c;
    }

    .stSelectbox label, .stSlider label, .stMultiSelect label {
        font-size: 0.78rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

HARM_CATEGORIES = [
    "harm_sexual",
    "harm_sexual_minors",
    "harm_harassment",
    "harm_harassment_threatening",
    "harm_hate",
    "harm_hate_threatening",
    "harm_violence",
    "harm_violence_graphic",
    "harm_self_harm",
    "harm_self_harm_intent",
    "harm_self_harm_instructions",
]
HARM_LABELS = [h.replace("harm_", "").replace("_", " ").title() for h in HARM_CATEGORIES]

SOURCE_LABELS = {
    "wildjailbreak": "WildJailbreak",
    "toxicchat": "Toxic-Chat",
    "trustairlab": "TrustAIRLab",
}
SOURCES = list(SOURCE_LABELS.values())

OWASP_CATEGORIES = ["LLM01", "LLM05", "LLM06", "LLM07", "LLM09", "LLM10"]
OWASP_LABELS = {
    "LLM01": "Prompt Injection",
    "LLM05": "Sensitive Info Disclosure",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM09": "Misinformation",
    "LLM10": "Unbounded Consumption",
}

TECHNIQUES = [f"T{str(i).zfill(2)}" for i in range(1, 13)]
TECHNIQUE_LABELS = {
    "T01": "Roleplay / Persona",
    "T02": "Hypothetical Framing",
    "T03": "Academic / Research",
    "T04": "Creative Writing",
    "T05": "Encoding / Obfuscation",
    "T06": "Translation",
    "T07": "Authority Override",
    "T08": "Emotional Manipulation",
    "T09": "Multi-turn Escalation",
    "T10": "Technical/Code Context",
    "T11": "Prompt Chaining",
    "T12": "Direct/Unframed Request",
}

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ASSETS_DIR = APP_DIR / "assets"
MASTER_PATH = DATA_DIR / "master_with_nlp.parquet"


def artifact_path(filename: str) -> Path:
    """Route an artifact filename to data/ or assets/ by extension.
    PNG and HTML render targets live under assets/; everything else (parquet,
    csv, md) lives under data/.
    """
    return (ASSETS_DIR if filename.endswith((".png", ".html")) else DATA_DIR) / filename  


def _add_harm_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Assign harm_quartile using qcut on scored rows only (harm_max > 0).
    Unscored rows get NaN — matches the NLP-analysis Stage 6B methodology.
    """
    if "harm_quartile" in df.columns:
        return df
    df["harm_quartile"] = pd.Series(pd.Categorical(
        [np.nan] * len(df),
        categories=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
        ordered=True,
    ), index=df.index)
    scored_mask = df["harm_max"] > 0
    if scored_mask.any():
        df.loc[scored_mask, "harm_quartile"] = pd.qcut(
            df.loc[scored_mask, "harm_max"], 4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
            duplicates="drop",
        )
    return df


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    """Load master_with_nlp.parquet as primary; fall back to master.parquet if NLP file is missing."""
    path = MASTER_PATH if MASTER_PATH.exists() else DATA_DIR / "master.parquet"
    if not path.exists():
        st.error("Neither master_with_nlp.parquet nor master.parquet found next to app.py.")
        st.stop()
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        st.stop()

    for col in HARM_CATEGORIES:
        if col not in df.columns:
            df[col] = np.nan
    if "harm_max" not in df.columns:
        df["harm_max"] = df[HARM_CATEGORIES].max(axis=1)
    if "source_dataset" in df.columns:
        df["source_dataset"] = df["source_dataset"].map(SOURCE_LABELS).fillna(df["source_dataset"])
    df = _add_harm_quartile(df)
    df.attrs["source_file"] = path.name
    return df


@st.cache_data(show_spinner=False)
def load_network_metrics() -> pd.DataFrame | None:
    p = artifact_path("network_metrics.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_lift_table() -> pd.DataFrame | None:
    p = artifact_path("lift_table.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    p = artifact_path("model_comparison_table.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_politeness_stats() -> pd.DataFrame | None:
    p = artifact_path("stats_table.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_nlp_findings_section(section_marker: str) -> str | None:
    """Return text of a named section from nlp_findings.md, or None."""
    p = artifact_path("nlp_findings.md")
    if not p.exists():
        return None
    text = p.read_text()
    # Sections split on '## ' h2 markers
    parts = text.split("\n## ")
    for part in parts:
        if part.strip().startswith(section_marker):
            return "## " + part.strip()
    return None


# Display-name mapping aligned with NLP-analysis stats_table.csv "category" column.
CATEGORY_DISPLAY = {
    "harm_sexual": "Sexual",
    "harm_sexual_minors": "Sexual / Minors",
    "harm_harassment": "Harassment",
    "harm_harassment_threatening": "Harassment (Threat.)",
    "harm_hate": "Hate",
    "harm_hate_threatening": "Hate (Threat.)",
    "harm_violence": "Violence",
    "harm_violence_graphic": "Violence (Graphic)",
    "harm_self_harm": "Self-harm",
    "harm_self_harm_intent": "Self-harm (Intent)",
    "harm_self_harm_instructions": "Self-harm (Instr.)",
}

# Per-category thresholds used by NLP-analysis Stage 6B script.
HIGH_HARM_THRESHOLD = 0.5
LOW_HARM_THRESHOLD = 0.3
MIN_HIGH_N = 15


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ GuardRail")
    st.markdown("<div style='font-size:0.75rem;color:#334155;margin-bottom:1.5rem'>Jailbreak Analysis Framework</div>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("### Filters")
    selected_techniques = st.multiselect(
        "Techniques",
        options=TECHNIQUES,
        default=TECHNIQUES,
        format_func=lambda t: f"{t}: {TECHNIQUE_LABELS[t]}",
        help=(
            "Jailbreak tactics the prompt uses. Each prompt is tagged with one primary "
            "technique (T01–T12). "
            "Deselect any to remove those prompts from every chart."
        ),
    )
    harm_threshold = st.slider(
        "Min harm_max", 0.0, 1.0, 0.0, 0.05,
        help=(
            "Keep only prompts whose worst-category score clears this threshold. "
            "harm_max = max across the 11 harm categories, so a prompt with one "
            "extreme category passes even if the other 10 are near-zero. "
            "Raise to 0.5 for 'moderation-flagged'; 0.9 for the severe tail."
        ),
    )
    selected_datasets = st.multiselect(
        "Source Datasets",
        options=SOURCES,
        default=SOURCES,
        help=(
            "Where the prompts came from. "
            "**WildJailbreak** (9,000): synthetic adversarial prompts from Allen AI; real tactics, GPT-4 phrasing. "
            "**Toxic-Chat** (1,152): real user inputs from the Vicuna demo; human-annotated. "
            "**TrustAIRLab** (1,202): hand-curated prompts from Reddit/Discord jailbreak communities."
        ),
    )

    st.divider()
    st.markdown("<div style='font-size:0.7rem;color:#475569'>Stages 1–6 complete ✓<br>Stage 10 (this app) in progress…<br>Stages 7–8 pending</div>",
                unsafe_allow_html=True)


# ── Data + Filtering ──────────────────────────────────────────────────────────

df_raw = load_data()

df = df_raw.copy()
if selected_techniques:
    df = df[df["technique_type"].isin(selected_techniques)]
if harm_threshold > 0:
    df = df[df["harm_max"] >= harm_threshold]
if selected_datasets and "source_dataset" in df.columns:
    df = df[df["source_dataset"].isin(selected_datasets)]

df_scored = df[df["harm_max"] > 0].copy()


# ── Header ────────────────────────────────────────────────────────────────────

st.title("GuardRail")
st.markdown("<div style='color:#334155;font-size:0.9rem;margin-bottom:1.5rem'>Jailbreak taxonomy, network analysis & harm prediction · GuardRail research project, Apr 2026</div>",
            unsafe_allow_html=True)

# Top-level KPIs
c1, c2, c3, c4 = st.columns(4)
for col, val, label in zip(
    [c1, c2, c3, c4],
    [f"{len(df):,}", f"{len(df_scored):,}",
     f"{df['harm_max'].mean():.3f}", f"{df['technique_type'].nunique()}"],
    ["Total Prompts", "Scored Rows", "Mean Harm Max", "Active Techniques"],
):
    col.markdown(f"""<div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "01 · Attack Surface Map",
    "02 · Technique × Harm Heatmap",
    "03 · Super-Attack Finder",
    "04 · Politeness Paradox",
    "05 · Topic Explorer",
    "06 · Risk Scorer",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 01 — Attack Surface Map  (live parts from parquet; network from network analysis)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Attack Surface Map")
    st.markdown(
        "<div style='color:#0f172a;font-size:0.95rem;margin-bottom:0.6rem'>"
        "<b>The question this tab answers:</b> Which jailbreak techniques are <i>structurally</i> "
        "most dangerous? Not just most common, but which sit at the centre of the attack graph, "
        "bridge between playbooks, or punch above their usage weight in harm? "
        "The answer informs which techniques a defender should block first."
        "</div>"
        "<div style='color:#334155;font-size:0.85rem;margin-bottom:1rem'>"
        "<b>How this tab builds up:</b> the <i>baseline attack surface</i> below is always live "
        "from the parquet. The <i>co-occurrence network, centrality metrics, and Louvain communities</i> "
        "come from the network analysis (Stage 7); sections auto-populate as those files arrive."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Network-analysis artifact status ─────────────────────────────────────
    expected_files = {
        "cooccurrence_graph.html": "Interactive pyvis network (hero view)",
        "network_metrics.csv": "PageRank, betweenness, community, harm-weighted PageRank",
        "playbook_communities.png": "Louvain community visualization",
        "cooccurrence_graph.png": "Static force-directed network",
        "defense_priority_matrix.png": "2×2 hub-residual × mean-harm quadrant",
        "network_findings.md": "Key findings summary",
    }
    present = {fn: artifact_path(fn).exists() for fn in expected_files}
    missing = [fn for fn, ok in present.items() if not ok]

    if missing:
        st.markdown(
            f"""<div class="stub-banner">
            ⏳ <b>Awaiting network analysis (Stage 7):</b> {len(missing)} / {len(expected_files)} files pending:
            {", ".join(f"<code>{fn}</code>" for fn in missing)}.
            Drop data files (<code>.csv</code>, <code>.md</code>) into <code>data/</code> and images / HTML into <code>assets/</code>; this tab will light up automatically.
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Baseline attack surface (from parquet — always live) ────────────────
    st.markdown("#### Baseline attack surface (from parquet)")
    st.markdown(
        "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.6rem'>"
        "Each bubble is one technique. <b>X-axis</b> = how often it appears in the corpus. "
        "<b>Y-axis</b> = mean harm_max of prompts using it. <b>Bubble size</b> = absolute count. "
        "Colour encodes mean harm (brighter = more harmful on average). "
        "Top-right bubbles are <i>frequent AND dangerous</i>: the raw-count priority signal. "
        "The network analysis adds the structural layer: which techniques are <i>hubs</i> beyond their raw frequency."
        "</div>",
        unsafe_allow_html=True,
    )
    tech_base = (
        df_scored.groupby("technique_type")
        .agg(count=("prompt_id", "size") if "prompt_id" in df_scored.columns else ("harm_max", "size"),
             mean_harm=("harm_max", "mean"))
        .reindex(TECHNIQUES)
        .fillna({"count": 0, "mean_harm": 0.0})
    )
    tech_base["label"] = [f"{t} · {TECHNIQUE_LABELS[t]}" for t in tech_base.index]
    fig_base = go.Figure(go.Scatter(
        x=tech_base["count"],
        y=tech_base["mean_harm"],
        mode="markers+text",
        text=tech_base.index,
        textposition="top center",
        textfont=dict(family="IBM Plex Mono", size=11, color="#0f172a"),
        marker=dict(
            size=np.clip(np.sqrt(tech_base["count"].values) * 2.2, 14, 60),
            color=tech_base["mean_harm"],
            colorscale=[[0, "#fff7ed"], [0.5, "#fdba74"], [1, "#9a3412"]],
            cmin=0, cmax=max(tech_base["mean_harm"].max(), 0.3),
            showscale=True,
            colorbar=dict(title=dict(text="Mean harm",
                                     font=dict(family="IBM Plex Mono", color="#334155")),
                          tickfont=dict(family="IBM Plex Mono", color="#334155")),
            line=dict(color="#ffffff", width=2),
            opacity=0.9,
        ),
        customdata=np.stack([tech_base["label"], tech_base["count"], tech_base["mean_harm"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Count: %{customdata[1]:,}<br>Mean harm: %{customdata[2]:.3f}<extra></extra>",
    ))
    fig_base.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font=dict(color="#334155", family="IBM Plex Sans"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                        font=dict(color="#0f172a", family="IBM Plex Sans", size=12)),
        xaxis=dict(title="Technique frequency (count of scored prompts)", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(title="Mean harm_max", showgrid=True, gridcolor="#e2e8f0", range=[0, None]),
        height=460,
        showlegend=False,
    )
    st.plotly_chart(fig_base, width="stretch")

    # ── Hero: interactive network (network analysis) ────────────────────────
    st.markdown("#### Co-occurrence network")
    graph_html = artifact_path("cooccurrence_graph.html")
    graph_png = artifact_path("cooccurrence_graph.png")
    community_png = artifact_path("playbook_communities.png")
    if graph_html.exists():
        st.markdown(
            "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.4rem'>"
            "Interactive pyvis render of the technique graph. Node size = PageRank; colour = Louvain community. "
            "Drag nodes to rearrange; hover for technique details."
            "</div>",
            unsafe_allow_html=True,
        )
        st.components.v1.html(graph_html.read_text(), height=640, scrolling=True)
    elif community_png.exists() or graph_png.exists():
        if community_png.exists():
            st.image(str(community_png), caption="Technique network coloured by Louvain community")
        if graph_png.exists():
            st.image(str(graph_png), caption="Technique co-occurrence network")
    else:
        st.info("Interactive network will render here once `cooccurrence_graph.html` is available from the network analysis.")

    # ── Centrality table (network analysis) ────────────────────────────────
    st.markdown("#### Centrality metrics")
    net_metrics = load_network_metrics()
    if net_metrics is not None:
        st.markdown(
            "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.4rem'>"
            "<b>PageRank</b> = overall influence. <b>Betweenness</b> = bridge role between communities. "
            "<b>Harm-weighted PageRank</b> = influence weighted by edge harm scores; this is the novel "
            "defender-priority metric. Sort by the column that matters for your question."
            "</div>",
            unsafe_allow_html=True,
        )
        display = net_metrics.copy()
        # Round numeric columns for cleaner display
        for c in display.select_dtypes(include="number").columns:
            display[c] = display[c].round(4)
        st.dataframe(display, width="stretch", hide_index=True)
    else:
        st.info("Centrality table will render here once `network_metrics.csv` is available from the network analysis.")

    # ── Defense priority matrix (network analysis 7E) ──────────────────────
    priority_png = artifact_path("defense_priority_matrix.png")
    if priority_png.exists():
        st.markdown("#### Defense priority matrix (hub residual × mean harm)")
        st.image(str(priority_png),
                 caption="Top-right quadrant = techniques more central than their frequency predicts AND high harm: block first.")

    # ── Narrative ──────────────────────────────────────────────────────────
    if artifact_path("network_findings.md").exists():
        text = artifact_path("network_findings.md").read_text()
        with st.expander("Network-analysis findings (from network_findings.md)", expanded=True):
            st.markdown(text)

    st.caption(
        "Key questions this tab will ultimately answer: (1) Which techniques are structural hubs? "
        "(2) Do techniques cluster into 3–5 meaningful playbooks (Social Engineering, Technical Obfuscation, "
        "Reframing, System Exploitation)? (3) Which techniques punch above their usage weight in harm?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 02 — Technique × Harm Heatmap  (LIVE — master.parquet only)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Technique × Harm Category Heatmap")
    st.markdown(
        "<div style='color:#0f172a;font-size:0.88rem;margin-bottom:0.4rem'>"
        "<b>How to read:</b> rows = techniques, columns = harm categories. Each cell is "
        "the chosen aggregate of that harm score among prompts using that technique. "
        "Brighter orange = the technique reliably drives that kind of harm; dark = it doesn't."
        "</div>"
        "<div style='color:#334155;font-size:0.78rem;margin-bottom:1rem'>"
        "Scored rows only (harm_max > 0). Sidebar's <i>Min harm_max</i> gates by severity; "
        "because harm_max = max across 11 categories, a single prompt typically only heats up one or two cells, "
        "so per-category means look low even when the filter is high. Switch Aggregation to "
        "<b>90th pct</b> to see worst-case intensity instead of the diluted average."
        "</div>",
        unsafe_allow_html=True,
    )

    col_opts, col_main = st.columns([1, 4])
    with col_opts:
        agg_method = st.radio(
            "Aggregation",
            ["Mean", "Median", "90th pct"],
            index=0,
            help=(
                "How to summarize harm scores across prompts of a given technique.\n\n"
                "• **Mean**: average score. Most prompts specialize in one harm category, "
                "so the mean gets diluted by zeros in the others.\n\n"
                "• **Median**: the typical prompt's score. Robust to outliers but often 0 "
                "for rare category/technique pairs.\n\n"
                "• **90th pct**: the score at the worst-case end. Cuts through dilution; "
                "best for spotting which technique × harm pairs are dangerous *when they hit*."
            ),
        )
        show_counts = st.checkbox(
            "Overlay row counts",
            value=False,
            help=(
                "Show the sample size (n) in each cell. Low n = noisy cell: T09 Multi-turn "
                "has only 33 rows vs T01 Roleplay's 4,415, so one prompt moves T09's numbers a lot."
            ),
        )
        sort_by = st.selectbox(
            "Sort techniques by",
            ["Frequency", "Mean harm_max", "Technique ID"],
            help=(
                "Row ordering of the heatmap.\n\n"
                "• **Frequency**: most-used techniques on top (attacker popularity).\n\n"
                "• **Mean harm_max**: most-harmful techniques on top (defender priority).\n\n"
                "• **Technique ID**: T01 → T12 numeric order (reference view)."
            ),
        )

    # Build pivot
    scored = df_scored.copy()
    if scored.empty:
        st.warning("No rows match current filters.")
    else:
        if agg_method == "Mean":
            pivot = scored.groupby("technique_type")[HARM_CATEGORIES].mean()
        elif agg_method == "Median":
            pivot = scored.groupby("technique_type")[HARM_CATEGORIES].median()
        else:
            pivot = scored.groupby("technique_type")[HARM_CATEGORIES].quantile(0.90)

        pivot.columns = HARM_LABELS
        pivot.index = [f"{t} · {TECHNIQUE_LABELS[t]}" for t in pivot.index]

        # Sort rows
        if sort_by == "Frequency":
            freq = scored["technique_type"].value_counts()
            freq.index = [f"{t} · {TECHNIQUE_LABELS[t]}" for t in freq.index]
            pivot = pivot.loc[pivot.index.intersection(freq.index)]
        elif sort_by == "Mean harm_max":
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        count_pivot = scored.groupby("technique_type").size()
        count_pivot.index = [f"{t} · {TECHNIQUE_LABELS[t]}" for t in count_pivot.index]

        # Annotation text
        if show_counts:
            ann_text = [[
                f"{pivot.loc[r, c]:.2f}<br><span style='font-size:8px'>n={count_pivot.get(r, 0)}</span>"
                for c in pivot.columns
            ] for r in pivot.index]
        else:
            ann_text = [[f"{v:.2f}" for v in row] for row in pivot.values]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=ann_text,
            texttemplate="%{text}",
            textfont=dict(size=10, family="IBM Plex Mono"),
            colorscale=[
                [0.0, "#f8fafc"],
                [0.25, "#ffedd5"],
                [0.5, "#fdba74"],
                [0.75, "#ea580c"],
                [1.0, "#9a3412"],
            ],
            zmin=0, zmax=1,
            colorbar=dict(
                title=dict(text="Score", font=dict(family="IBM Plex Mono", color="#334155")),
                tickfont=dict(family="IBM Plex Mono", color="#334155"),
                bgcolor="#ffffff",
                bordercolor="#e2e8f0",
            ),
            hoverongaps=False,
        ))
        fig_heat.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="IBM Plex Sans", color="#334155"),
            xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=10, family="IBM Plex Mono")),
            margin=dict(l=240, b=120, t=40, r=40),
            height=480,
        )
        with col_main:
            st.plotly_chart(fig_heat, width="stretch")

        # Summary bullets
        st.divider()
        top_pair = pivot.stack().idxmax()
        low_pair = pivot.stack().idxmin()
        st.markdown(f"""
        **Quick reads:**
        - Highest mean score: **{top_pair[0]}** × **{top_pair[1]}** ({pivot.loc[top_pair]:.3f})
        - Lowest mean score: **{low_pair[0]}** × **{low_pair[1]}** ({pivot.loc[low_pair]:.3f})
        - Rows in view: **{len(scored):,}** scored prompts across **{scored['technique_type'].nunique()}** techniques
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 03 — Super-Attack Finder  (single-technique leaderboard live;
#                                pair-lift heatmap from network analysis)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Super-Attack Finder")
    st.markdown(
        "<div style='color:#0f172a;font-size:0.95rem;margin-bottom:0.6rem'>"
        "<b>The question this tab answers:</b> Which technique <i>combinations</i> punch above their "
        "weight on a specific harm? For each harm category, a super-attack is a technique pair whose "
        "prompts score far higher than the overall corpus average in that category, via the "
        "<b>lift</b> metric: mean-of-pair ÷ mean-of-corpus."
        "</div>"
        "<div style='color:#334155;font-size:0.85rem;margin-bottom:1rem'>"
        "<b>How this tab builds up:</b> the <i>per-harm technique leaderboard</i> below is live from "
        "the parquet; it shows which single techniques lead in each harm category. The true "
        "<i>pair-lift heatmap</i> (super-attacks proper) comes from the network analysis (Stage 7) and "
        "auto-populates once the files arrive."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Artifact status ────────────────────────────────────────────────────
    expected_files_t3 = {
        "super_attack_heatmap.png": "Pair-lift heatmap faceted by harm category",
    }
    missing_t3 = [fn for fn in expected_files_t3 if not artifact_path(fn).exists()]
    if missing_t3:
        st.markdown(
            f"""<div class="stub-banner">
            ⏳ <b>Awaiting network analysis (Stage 7):</b> {len(missing_t3)} / {len(expected_files_t3)} files pending:
            {", ".join(f"<code>{fn}</code>" for fn in missing_t3)}.
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Harm-category selector (drives the live leaderboard below) ─────────
    harm_options = [(c, CATEGORY_DISPLAY[c]) for c in HARM_CATEGORIES if c in df_scored.columns]
    display_to_col = {label: col for col, label in harm_options}
    selected_label = st.selectbox(
        "Harm category to focus on",
        [label for _, label in harm_options],
        index=[label for _, label in harm_options].index("Hate") if any(l == "Hate" for _, l in harm_options) else 0,
        help=(
            "Sets the harm category for the single-technique leaderboard below. "
            "Super-attacks are category-specific: the pair that drives Hate isn't the same pair that "
            "drives Self-harm. The static pair-lift PNG further down already facets by category, so "
            "this selector doesn't affect it."
        ),
    )
    selected_col = display_to_col[selected_label]

    # ── Live: single-technique leaderboard for the selected harm ──────────
    st.markdown(f"#### Single-technique leaderboard · {selected_label}")
    st.markdown(
        "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.6rem'>"
        "For the selected harm category: mean and 90th-percentile harm score per technique, "
        "filtered to techniques with at least 30 scored prompts. Mean = typical severity. "
        "90th pct = worst-case severity. Large gap = technique is inconsistent; dangerous when it lands, "
        "but not always. This sets the single-technique baseline; super-attack pairs should exceed both."
        "</div>",
        unsafe_allow_html=True,
    )
    per_tech = (
        df_scored.groupby("technique_type")[selected_col]
        .agg(["count", "mean", lambda s: s.quantile(0.90)])
    )
    per_tech.columns = ["count", "mean", "p90"]
    per_tech = per_tech[per_tech["count"] >= 30].sort_values("mean", ascending=True)
    if per_tech.empty:
        st.info("No technique has ≥30 scored prompts under current filters.")
    else:
        per_tech["label"] = [f"{t} · {TECHNIQUE_LABELS[t]}" for t in per_tech.index]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Mean", x=per_tech["mean"], y=per_tech["label"],
            orientation="h", marker_color="#f97316",
            customdata=per_tech["count"],
            hovertemplate="<b>%{y}</b><br>Mean: %{x:.3f}<br>n = %{customdata:,}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            name="90th pct", x=per_tech["p90"], y=per_tech["label"],
            orientation="h", marker_color="#9a3412",
            customdata=per_tech["count"],
            hovertemplate="<b>%{y}</b><br>90th pct: %{x:.3f}<br>n = %{customdata:,}<extra></extra>",
        ))
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(color="#334155", family="IBM Plex Sans"),
            legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0"),
            hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                            font=dict(color="#0f172a", family="IBM Plex Sans", size=12)),
            xaxis=dict(title=f"Harm score ({selected_label})", range=[0, 1], showgrid=True, gridcolor="#e2e8f0"),
            yaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11)),
            height=max(320, 28 * len(per_tech) + 80),
            margin=dict(l=220, r=40, t=20, b=50),
        )
        st.plotly_chart(fig_bar, width="stretch")

        # Corpus baseline for lift calibration
        baseline = df_scored[selected_col].mean()
        st.caption(
            f"Corpus-wide mean for **{selected_label}** across all scored prompts: **{baseline:.3f}**. "
            "The network analysis's super-attack pairs will show technique pairs whose mean exceeds this by 1.5× or more."
        )

    # ── Hero: pair-lift heatmap (network analysis) ─────────────────────────
    st.markdown("#### Pair-lift heatmap (super-attacks)")
    super_png = artifact_path("super_attack_heatmap.png")
    if super_png.exists():
        st.markdown(
            "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.4rem'>"
            "Rows = T_i, columns = T_j. Each cell = lift (pair mean ÷ corpus mean), faceted by harm category. "
            "Bright cells (lift &gt; 1.5) are super-attacks: the pair amplifies that specific harm beyond "
            "what either technique does alone. The harm-category selector above is for the single-technique "
            "leaderboard; the network-analysis render below shows all categories at once."
            "</div>",
            unsafe_allow_html=True,
        )
        st.image(str(super_png),
                 caption="Super-attack heatmap, faceted by harm category (Stage 7)")
    else:
        st.info("Pair-lift heatmap will render here once `super_attack_heatmap.png` is available from the network analysis.")

    # ── Narrative (shared with Tab 01) ────────────────────────────────────
    if artifact_path("network_findings.md").exists():
        text = artifact_path("network_findings.md").read_text()
        with st.expander("Network-analysis findings (from network_findings.md)"):
            st.markdown(text)

    st.caption(
        "Defender framing: lift &gt; 1.5 on a specific harm means blocking the pair is "
        "dramatically more valuable than blocking either technique alone. Different harm categories "
        "typically have different super-attacks; that's the 'attack recipes vary' finding."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 04 — Politeness Paradox  (LIVE)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Politeness Paradox")

    if "politeness_score" not in df_scored.columns or "sentiment_compound" not in df_scored.columns:
        st.warning(
            "`politeness_score` / `sentiment_compound` not in the current dataset. "
            f"loaded `{df_raw.attrs.get('source_file', MASTER_PATH.name)}`. "
            "Switch to `master_with_nlp.parquet` to enable this tab."
        )
    else:
        st.markdown(
            "<div style='color:#0f172a;font-size:0.95rem;margin-bottom:0.6rem'>"
            "<b>The question this tab answers:</b> do the <i>worst</i> jailbreaks actually "
            "sound <i>polite</i>? Intuition says harmful prompts should sound angry or aggressive. "
            "But attackers often wrap malicious requests in please/thank-you/I'd-appreciate "
            "framing to slip past tone-based filters. <b>This tab tests that paradox.</b>"
            "</div>"
            "<div style='color:#334155;font-size:0.85rem;margin-bottom:0.6rem'>"
            "<b>How to read the violin plots:</b> each shape is the distribution of scores for "
            "prompts in one <b>harm quartile</b>: Q1 is the least-harmful 25% of prompts, Q4 "
            "is the most harmful. A wider section of the violin means more prompts sit at that "
            "score. The box inside shows the middle 50%; the white dot is the median. "
            "<b>If the paradox holds, Q4 sits higher on the politeness axis than Q1.</b>"
            "</div>"
            "<div style='color:#334155;font-size:0.78rem;margin-bottom:1rem'>"
            f"{len(df_scored):,} scored rows · sidebar filters applied · "
            "<b>politeness_score</b> = rule-based count of polite tokens (please, thanks, could, would…) "
            "per 100 words · <b>sentiment_compound</b> = VADER emotional valence, −1 (angry/negative) "
            "to +1 (warm/positive)."
            "</div>",
            unsafe_allow_html=True,
        )

        c_left, c_right = st.columns(2)
        for col, metric, y_label, title, caption in [
            (c_left,  "politeness_score",  "Politeness Score",
             "Politeness Score by Harm Quartile",
             "Look for Q4 sitting *above* Q1. That's the paradox in action."),
            (c_right, "sentiment_compound", "VADER Compound",
             "Sentiment by Harm Quartile",
             "Look for Q4 sitting *below* Q1. Finding: high-harm prompts trend more negative, not more positive."),
        ]:
            fig_v = px.violin(
                df_scored, x="harm_quartile", y=metric,
                color="harm_quartile",
                color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#f97316"],
                category_orders={"harm_quartile": ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]},
                box=True, points=False,
                labels={"harm_quartile": "Harm Quartile", metric: y_label},
                title=title,
            )
            fig_v.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#334155", family="IBM Plex Sans"),
                showlegend=False,
                height=400,
            )
            col.plotly_chart(fig_v, width="stretch")
            col.caption(caption)

        # Per-category breakdown
        st.markdown("#### Per-category politeness breakdown")
        st.markdown(
            f"<div style='color:#334155;font-size:0.82rem;margin-bottom:0.8rem'>"
            f"For each harm category, split prompts into <b>high</b> (score &gt; {HIGH_HARM_THRESHOLD}) "
            f"and <b>low</b> (score &lt; {LOW_HARM_THRESHOLD}), then compare their average politeness. "
            f"Categories with fewer than {MIN_HIGH_N} high-harm prompts are dropped (too noisy to compare). "
            f"Thresholds and filter match NLP-analysis Stage 6B analysis."
            "</div>",
            unsafe_allow_html=True,
        )

        cat_rows = []
        for harm_col in HARM_CATEGORIES:
            if harm_col not in df_scored.columns:
                continue
            hi_mask = df_scored[harm_col] > HIGH_HARM_THRESHOLD
            lo_mask = df_scored[harm_col] < LOW_HARM_THRESHOLD
            n_hi = int(hi_mask.sum())
            if n_hi < MIN_HIGH_N:
                continue  # skip low-n categories (matches NLP-analysis filter)
            cat_rows.append({
                "Category": CATEGORY_DISPLAY.get(harm_col, harm_col),
                "n (high)": n_hi,
                "n (low)": int(lo_mask.sum()),
                "High-harm": df_scored.loc[hi_mask, "politeness_score"].mean(),
                "Low-harm":  df_scored.loc[lo_mask, "politeness_score"].mean(),
            })

        if cat_rows:
            cat_df = pd.DataFrame(cat_rows)
            cat_df["Delta"] = cat_df["High-harm"] - cat_df["Low-harm"]
            cat_df = cat_df.sort_values("Delta", ascending=False)

            fig_cat = go.Figure()
            fig_cat.add_trace(go.Bar(
                name=f"High-harm (>{HIGH_HARM_THRESHOLD})",
                x=cat_df["Category"], y=cat_df["High-harm"],
                marker_color="#f97316",
                customdata=cat_df["n (high)"],
                hovertemplate="<b>%{x}</b><br>Mean politeness (high-harm): %{y:.3f}<br>n = %{customdata}<extra></extra>",
            ))
            fig_cat.add_trace(go.Bar(
                name=f"Low-harm (<{LOW_HARM_THRESHOLD})",
                x=cat_df["Category"], y=cat_df["Low-harm"],
                marker_color="#3b82f6",
                customdata=cat_df["n (low)"],
                hovertemplate="<b>%{x}</b><br>Mean politeness (low-harm): %{y:.3f}<br>n = %{customdata}<extra></extra>",
            ))
            fig_cat.update_layout(
                barmode="group",
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#334155", family="IBM Plex Sans"),
                legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0"),
                hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                                font=dict(color="#0f172a", family="IBM Plex Sans", size=12)),
                xaxis=dict(tickangle=-30, categoryorder="array", categoryarray=cat_df["Category"].tolist()),
                yaxis=dict(title="Mean politeness score"),
                height=380,
            )
            st.plotly_chart(fig_cat, width="stretch")

            # Statistical test results from NLP-analysis stats_table.csv
            stats_df = load_politeness_stats()
            if stats_df is not None:
                st.markdown("#### Statistical test results (from NLP-analysis `stats_table.csv`)")
                st.markdown(
                    "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.6rem'>"
                    "Mann-Whitney U test comparing high-harm vs low-harm prompts per category. "
                    "<b>p-value</b> = probability of seeing this difference by chance (lower = more real). "
                    "<b>Effect size r</b> = rank-biserial correlation, −1 to +1 "
                    "(positive ⇒ high-harm more polite; negative ⇒ less polite). "
                    "Significance flags: ✅ p&lt;0.05, ⚠️ marked if n_high&lt;30 (directional only)."
                    "</div>",
                    unsafe_allow_html=True,
                )

                def _sig(p, n_high, small_flag):
                    if pd.isna(p):
                        return "n/a"
                    mark = "✅" if p < 0.05 else "·"
                    caveat = " ⚠️" if (small_flag or n_high < 30) else ""
                    return f"{mark}{caveat}"

                display = pd.DataFrame({
                    "Category": stats_df["category"],
                    "n high": stats_df["n_high"].astype(int),
                    "n low": stats_df["n_low"].astype(int),
                    "Politeness U": stats_df["politeness_U"].round(0).astype("Int64"),
                    "Politeness p": stats_df["politeness_p"].apply(lambda x: f"{x:.3g}" if pd.notna(x) else "n/a"),
                    "Politeness r": stats_df["politeness_r"].round(3),
                    "Pol. sig.": [_sig(p, n, f)
                                  for p, n, f in zip(stats_df["politeness_p"],
                                                     stats_df["n_high"],
                                                     stats_df["small_sample_flag"])],
                    "Sentiment p": stats_df["sentiment_p"].apply(lambda x: f"{x:.3g}" if pd.notna(x) else "n/a"),
                    "Sentiment r": stats_df["sentiment_r"].round(3),
                    "Sent. sig.": [_sig(p, n, f)
                                   for p, n, f in zip(stats_df["sentiment_p"],
                                                      stats_df["n_high"],
                                                      stats_df["small_sample_flag"])],
                })
                st.dataframe(display, width="stretch", hide_index=True)

            # Narrative bullets straight from NLP-analysis nlp_findings.md
            findings_6b = load_nlp_findings_section("6B")
            if findings_6b:
                with st.expander("NLP-analysis findings (from nlp_findings.md §6B)", expanded=True):
                    st.markdown(findings_6b)

    st.caption(
        "Key finding (from §6B): the paradox does NOT hold globally. High-harm prompts are "
        "LESS polite (Q4 median=0.00 vs Q1=0.51, p=0.0000) and more emotionally negative "
        "(VADER Q4=0.616 vs Q1=0.883). It emerges only in Self-harm categories, "
        "and those have small samples (treat as directional). "
        "Implication for defenders: a global tone-based filter would miss the signal."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 05 — Topic Explorer  (LIVE — lda_topic in parquet + NLP-analysis artifacts)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Topic Explorer")

    if "lda_topic" not in df.columns:
        st.warning("`lda_topic` column not in current dataset. Switch to `master_with_nlp.parquet`.")
    else:
        st.markdown(
            "<div style='color:#0f172a;font-size:0.95rem;margin-bottom:0.6rem'>"
            "<b>The question this tab answers:</b> does a blind topic model (LDA), given "
            "nothing but the raw text, independently rediscover the T01–T12 taxonomy? "
            "If yes, the taxonomy reflects genuine linguistic structure, not something we imposed."
            "</div>"
            "<div style='color:#334155;font-size:0.85rem;margin-bottom:0.6rem'>"
            "<b>How to read:</b> LDA was trained with <b>k=6</b> topics (chosen from {6,8,10} by "
            "coherence score). Each prompt was assigned its single dominant topic. The charts below "
            "show how these 6 topics map onto the 12 techniques and the 11 harm categories."
            "</div>"
            "<div style='color:#334155;font-size:0.78rem;margin-bottom:1rem'>"
            f"{int((df['lda_topic'].notna()).sum()):,} prompts with topic assignments "
            f"({int(df['lda_topic'].isna().sum())} NaN dropped) · sidebar filters applied."
            "</div>",
            unsafe_allow_html=True,
        )

        lda_df = df.dropna(subset=["lda_topic"]).copy()
        lda_df["lda_topic"] = lda_df["lda_topic"].astype(int)
        topic_labels = [f"Topic {t}" for t in sorted(lda_df["lda_topic"].unique())]

        # ── Interactive pyLDAvis (NLP-analysis primary artifact) ─────────────────
        st.markdown("#### Interactive topic model (pyLDAvis)")
        st.markdown(
            "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.6rem'>"
            "Left panel: each circle is a topic; distance ≈ inter-topic similarity; area ≈ topic prevalence. "
            "Right panel: top words, with the red bar showing word-in-topic frequency vs the blue bar "
            "(word-in-corpus). Drag <code>λ</code> up to rank words by topic-specificity, down to rank by overall frequency."
            "</div>",
            unsafe_allow_html=True,
        )
        lda_html_path = artifact_path("lda_topics.html")
        if lda_html_path.exists():
            st.components.v1.html(lda_html_path.read_text(), height=720, scrolling=True)
        else:
            st.info("Drop `lda_topics.html` next to app.py to enable the interactive view.")

        # ── Topic × Technique crosstab ─────────────────────────────────────────
        st.markdown("#### Topic × Technique")
        st.markdown(
            "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.6rem'>"
            "Row-normalised proportion: within each topic, what fraction of prompts carry each technique label? "
            "<b>A row dominated by one technique is a topic that aligns with that tactic.</b> "
            "This is the direct test of taxonomy validity."
            "</div>",
            unsafe_allow_html=True,
        )
        topic_tech = pd.crosstab(lda_df["lda_topic"], lda_df["technique_type"], normalize="index")
        topic_tech.index = [f"Topic {t}" for t in topic_tech.index]
        # Ensure all techniques in column order, even if some absent from topic
        for t in TECHNIQUES:
            if t not in topic_tech.columns:
                topic_tech[t] = 0.0
        topic_tech = topic_tech[TECHNIQUES]

        fig_tt = go.Figure(go.Heatmap(
            z=topic_tech.values,
            x=[f"{t} · {TECHNIQUE_LABELS[t]}" for t in topic_tech.columns],
            y=topic_tech.index.tolist(),
            text=[[f"{v:.0%}" if v >= 0.01 else "" for v in row] for row in topic_tech.values],
            texttemplate="%{text}",
            textfont=dict(size=10, family="IBM Plex Mono"),
            colorscale=[[0, "#f8fafc"], [0.3, "#ffedd5"], [0.6, "#fdba74"], [1, "#9a3412"]],
            zmin=0, zmax=topic_tech.values.max(),
            colorbar=dict(
                title=dict(text="Share", font=dict(family="IBM Plex Mono", color="#334155")),
                tickfont=dict(family="IBM Plex Mono", color="#334155"),
                tickformat=".0%",
            ),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Share: %{z:.1%}<extra></extra>",
        ))
        fig_tt.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(family="IBM Plex Sans", color="#334155"),
            hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                            font=dict(color="#0f172a", family="IBM Plex Sans", size=12)),
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=11, family="IBM Plex Mono")),
            margin=dict(l=80, b=140, t=20, r=20),
            height=360,
        )
        st.plotly_chart(fig_tt, width="stretch")

        # Auto-interpret: dominant technique per topic
        dom_rows = []
        for topic_row in topic_tech.index:
            row = topic_tech.loc[topic_row]
            top_tech = row.idxmax()
            share = row.max()
            dom_rows.append({
                "Topic": topic_row,
                "Dominant technique": f"{top_tech} · {TECHNIQUE_LABELS[top_tech]}",
                "Share": f"{share:.1%}",
                "Prompts in topic": int((lda_df["lda_topic"] == int(topic_row.split()[-1])).sum()),
            })
        st.markdown("**Dominant technique per topic:**")
        st.dataframe(pd.DataFrame(dom_rows), width="stretch", hide_index=True)

        # ── Topic × Harm crosstab ──────────────────────────────────────────────
        st.markdown("#### Topic × Harm Category")
        st.markdown(
            "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.6rem'>"
            "Mean harm score within each (topic, harm category) cell, among <b>scored rows</b> "
            "(harm_max &gt; 0). Brighter = that topic consistently produces prompts scoring high in that harm."
            "</div>",
            unsafe_allow_html=True,
        )
        lda_scored = lda_df[lda_df["harm_max"] > 0]
        topic_harm = lda_scored.groupby("lda_topic")[HARM_CATEGORIES].mean()
        topic_harm.index = [f"Topic {t}" for t in topic_harm.index]
        topic_harm.columns = [CATEGORY_DISPLAY.get(c, c) for c in topic_harm.columns]

        fig_th = go.Figure(go.Heatmap(
            z=topic_harm.values,
            x=topic_harm.columns.tolist(),
            y=topic_harm.index.tolist(),
            text=[[f"{v:.2f}" for v in row] for row in topic_harm.values],
            texttemplate="%{text}",
            textfont=dict(size=10, family="IBM Plex Mono"),
            colorscale=[[0, "#f8fafc"], [0.3, "#ffedd5"], [0.6, "#fdba74"], [1, "#9a3412"]],
            zmin=0, zmax=max(topic_harm.values.max(), 0.1),
            colorbar=dict(
                title=dict(text="Mean", font=dict(family="IBM Plex Mono", color="#334155")),
                tickfont=dict(family="IBM Plex Mono", color="#334155"),
            ),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Mean score: %{z:.3f}<extra></extra>",
        ))
        fig_th.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            font=dict(family="IBM Plex Sans", color="#334155"),
            hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                            font=dict(color="#0f172a", family="IBM Plex Sans", size=12)),
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=11, family="IBM Plex Mono")),
            margin=dict(l=80, b=140, t=20, r=20),
            height=360,
        )
        st.plotly_chart(fig_th, width="stretch")

        # ── NLP-analysis static PNG versions (fallback / alternate rendering) ────
        png_t = artifact_path("topic_technique_matrix.png")
        png_h = artifact_path("topic_harm_matrix.png")
        if png_t.exists() or png_h.exists():
            with st.expander("Static PNG versions (NLP-analysis rendering)"):
                if png_t.exists():
                    st.image(str(png_t), caption="NLP-analysis topic × technique matrix")
                if png_h.exists():
                    st.image(str(png_h), caption="NLP-analysis topic × harm matrix")

        # ── Narrative bullet from nlp_findings.md §6C ──────────────────────────
        findings_6c = load_nlp_findings_section("6C")
        if findings_6c:
            with st.expander("NLP-analysis findings", expanded=True):
                st.markdown(findings_6c)

    st.caption(
        "Key finding (from §6C): 5 of 6 LDA topics have a single dominant technique at ≥40% membership. "
        "The taxonomy is structurally real, not imposed; LDA found it blind."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 06 — Jailbreak Risk Scorer  (loaders for risk-model Stage 8 deliverables)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### Jailbreak Risk Scorer")
    st.markdown(
        "<div style='color:#0f172a;font-size:0.95rem;margin-bottom:0.6rem'>"
        "<b>The question this tab answers:</b> does layering the taxonomy (technique, OWASP) and "
        "network features (PageRank, community) onto a text model actually <i>improve</i> harm "
        "prediction? This is the 'the analysis is useful' proof: the slide-8 headline is the "
        "<i>delta</i> in performance, not the absolute score."
        "</div>"
        "<div style='color:#334155;font-size:0.85rem;margin-bottom:1rem'>"
        "<b>What will live here when the risk model lands:</b> (1) the layered model-comparison chart "
        "(text → +taxonomy → +network), (2) the per-category performance bars, "
        "(3) feature importance, (4) the findings narrative."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Artifact status ────────────────────────────────────────────────────
    expected_files_t6 = {
        "model_comparison_table.csv": "Layered metrics: text / +taxonomy / +network × harm category",
        "per_category_performance.png": "Per-harm-category model performance bars",
        "feature_importance.png": "Top-30 Ridge coefficients for the network-augmented model",
        "prediction_findings.md": "Key findings summary (delta framing)",
    }
    missing_t6 = [fn for fn in expected_files_t6 if not artifact_path(fn).exists()]
    if missing_t6:
        st.markdown(
            f"""<div class="stub-banner">
            ⏳ <b>Awaiting risk model (Stage 8):</b> {len(missing_t6)} / {len(expected_files_t6)} files pending:
            {", ".join(f"<code>{fn}</code>" for fn in missing_t6)}.
            Drop data files (<code>.csv</code>, <code>.md</code>) into <code>data/</code> and images / HTML into <code>assets/</code>; this tab will light up automatically.
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Model comparison table (risk model) ────────────────────────────────
    st.markdown("#### Layered model comparison")
    model_cmp = load_model_comparison()
    if model_cmp is not None:
        st.markdown(
            "<div style='color:#334155;font-size:0.82rem;margin-bottom:0.4rem'>"
            "Each row compares the same harm target across three feature tiers. "
            "<b>Headline is the delta</b>: how much does adding taxonomy (+tax) on top of text, "
            "and adding network (+net) on top of that, move the metric?"
            "</div>",
            unsafe_allow_html=True,
        )
        display_cmp = model_cmp.copy()
        for c in display_cmp.select_dtypes(include="number").columns:
            display_cmp[c] = display_cmp[c].round(4)
        st.dataframe(display_cmp, width="stretch", hide_index=True)
    else:
        st.info("Model comparison table will render here once `model_comparison_table.csv` is available from the risk model.")

    # ── Per-category performance (risk model) ──────────────────────────────
    perf_png = artifact_path("per_category_performance.png")
    feat_png = artifact_path("feature_importance.png")
    if perf_png.exists() or feat_png.exists():
        st.markdown("#### Diagnostic plots")
        if perf_png.exists():
            st.image(str(perf_png),
                     caption="Per-category performance: text / +taxonomy / +network")
        if feat_png.exists():
            st.image(str(feat_png),
                     caption="Top Ridge coefficients from the network-augmented model")

    # ── Narrative ──────────────────────────────────────────────────────────
    if artifact_path("prediction_findings.md").exists():
        text = artifact_path("prediction_findings.md").read_text()
        with st.expander("Risk-model findings (from prediction_findings.md)", expanded=True):
            st.markdown(text)

    st.caption(
        "Slide-8 framing (per work plan): lead with the increment: "
        "'adding taxonomy features improved AUC by 0.07 on average; network features added another 0.03.' "
        "That directly proves the analytical framework adds predictive value."
    )