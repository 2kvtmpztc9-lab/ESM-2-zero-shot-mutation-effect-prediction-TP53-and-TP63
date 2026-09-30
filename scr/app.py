import os
os.environ['STREAMLIT_DISABLE_TORCH_MODULE_WATCHING'] = '1'

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="ESM 2: TP53 mutation effects", layout="wide")

#Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

st.title("Zero-shot prediction of TP53 mutation effects with ESM-2")
st.markdown(""" 
Tool based on the protein language model **ESM-2** (Meta, 650M parameters).
Method: **masked marginal scoring** - score = log P(mutant | context) - log P(wild type | context).

**Negative score**, it mean the model considers the substitution unlikely (potentially pathogenic).
""")

@st.cache_data
def load_scores():
    return pd.read_csv(os.path.join(DATA, "tp53_scores.csv"))

@st.cache_data
def load_validation():
    try:
        return pd.read_csv(os.path.join(DATA, "validation_650m.csv"))
    except FileNotFoundError:
        return None

df = load_scores()
val = load_validation()

# Section 1. Validation
st.header("1. Validation on known TP53 mutations")
if val is not None:
    from sklearn.metrics import roc_auc_score
    val["true"] = (val["label"] == "pathogenic").astype(int)
    auc = roc_auc_score(val["true"], -val["score"])

    fig_val = px.strip(
        val, x="score", y="label", color="label",
        color_discrete_map={"pathogenic": "#d62728", "benign": "#1f77b4"},
        labels={"score": "Score (ESM-2 650M)", "label": "Class"},
        hover_data=["mutation", "pos", "wt", "mut"]
    )
    fig_val.add_vline(x=-4, line_dash="dash", line_color="gray",
                      annotation_text="Threshold -4", annotation_position="top")
    st.plotly_chart(fig_val, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean score (pathogenic)",
                f"{val[val.label=='pathogenic'].score.mean():.2f}")
    col2.metric("Mean score (benign)",
                f"{val[val.label=='benign'].score.mean():.2f}")
    col3.metric("ROC-AUC", f"{auc:.3f}")

    st.caption(f"Total mutations: {len(val)} "
               f"(pathogenic: {(val.label=='pathogenic').sum()}, "
               f"benign: {(val.label=='benign').sum()})")
else:
    st.warning("File validation_650m.csv not found.")

#Section 2. Heatmap
st.header("2. Heatmap of all TP53 mutations (model t12_35M)")
st.caption("Full scan of 7467 missense mutations. "
           "Shows which protein positions are sensitive to substitutions.")

pos_min, pos_max = st.slider(
    "Position range",
    int(df["pos"].min()), int(df["pos"].max()),
    (1, 120)
)
filtered = df[(df["pos"] >= pos_min) & (df["pos"] <= pos_max)]

pivot = filtered.pivot_table(index="mut", columns="pos", values="score", aggfunc="first")
fig_hm = px.imshow(
    pivot,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    labels={"x": "Position", "y": "Mutant aa", "color": "Score"},
    zmin=-8, zmax=2
)
st.plotly_chart(fig_hm, use_container_width=True)

#Section 3. Top pathogenic
st.header("3. Top predicted pathogenic mutations")
top_n = st.slider("Number to display", 5, 50, 20)
top = filtered.nsmallest(top_n, "score")[["pos", "wt", "mut", "score"]].reset_index(drop=True)
st.dataframe(top, use_container_width=True)

st.markdown("""
---
### Limitations
- The `t12_35M` model used for the full scan gives a weak signal; 
  validation was performed with `t33_650M`.
- The start methionine (position 1) yields artifactually low scores.
- The method performs worse on the C-terminal regulatory region of TP53 (positions 335–347).
- This zero-shot approach does not replace experimental validation.
""")
