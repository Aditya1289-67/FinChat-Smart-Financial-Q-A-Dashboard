
import re
import io
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# -------------------------- CONFIG -----------------------------
st.set_page_config(page_title="FinChat — Q&A + Dashboard", page_icon="💹", layout="wide")

st.title("💹 FinChat — Smart Q&A + Visual Dashboard")
st.caption("Upload structured financial CSV and ask questions or explore via the dashboard.")

# -------------------------- HELPERS ----------------------------

METRIC_MAP = {
    "revenue": "Total Revenue (Billion $)",
    "total revenue": "Total Revenue (Billion $)",
    "net income": "Net Income (Billion $)",
    "income": "Net Income (Billion $)",
    "profits": "Net Income (Billion $)",
    "assets": "Total Assets (Billion $)",
    "total assets": "Total Assets (Billion $)",
    "liabilities": "Total Liabilities (Billion $)",
    "total liabilities": "Total Liabilities (Billion $)",
    "cash flow": "Cash Flow from Operating Activities (Billion $)",
    "operating cash flow": "Cash Flow from Operating Activities (Billion $)",
}


def safe_div(a, b):
    try:
        if b == 0 or b is None or a is None:
            return None
        return a / b
    except Exception:
        return None


def list_companies(df: pd.DataFrame) -> List[str]:
    return sorted(df["Company"].dropna().unique().tolist())


def list_years(df: pd.DataFrame) -> List[int]:
    return sorted(pd.to_numeric(df["Fiscal Year"], errors="coerce").dropna().astype(int).unique().tolist())


def parse_years_in_text(text: str, available_years: List[int]) -> List[int]:
    text_low = text.lower()
    found = []
    # explicit years
    for y in available_years:
        if str(y) in text_low:
            found.append(y)
    # ranges like 2018-2022
    ranges = re.findall(r"(\b\d{4})\s*[-–to]{1,3}\s*(\d{4}\b)", text_low)
    for a, b in ranges:
        try:
            a_i, b_i = int(a), int(b)
            for y in range(a_i, b_i + 1):
                if y in available_years and y not in found:
                    found.append(y)
        except:
            pass
    # last N years
    m = re.search(r"last\s+(\d+)\s+years?", text_low)
    if m:
        n = int(m.group(1))
        av = sorted(available_years)
        tail = av[-n:]
        for y in tail:
            if y not in found:
                found.append(y)
    return sorted(found)


def detect_companies_in_text(text: str, companies: List[str]) -> List[str]:
    text_low = text.lower()
    found = []
    for c in companies:
        if c.lower() in text_low:
            found.append(c)
    return found


def detect_metric(text: str) -> Optional[str]:
    text_low = text.lower()
    for k, v in METRIC_MAP.items():
        if k in text_low:
            return v
    # fallback: look for words like "margin" or "ratio"
    if "ratio" in text_low:
        return "ratio"
    return None


def is_growth_query(text: str) -> bool:
    return "growth" in text.lower() or "increase" in text.lower() or "grew" in text.lower()


def is_compare_query(text: str) -> bool:
    return "compare" in text.lower() or " vs " in text.lower() or "versus" in text.lower()


def format_billion(val) -> str:
    try:
        if pd.isna(val):
            return "N/A"
        return f"{val:.2f} B$"
    except:
        return str(val)

# -------------------------- LOAD DATA --------------------------

uploaded = st.file_uploader("Upload your financial CSV (structured)", type=["csv"], key="main_csv")
if uploaded is None:
    st.info("Please upload the CSV you provided earlier (financial_data.csv).")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Basic validation
required_cols = ["Company", "Fiscal Year"]
if not all(c in df.columns for c in required_cols):
    st.error(f"CSV must contain columns: {required_cols}")
    st.stop()

# normalize numeric columns (try to coerce)
for col in df.columns:
    if col not in ["Company", "Fiscal Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

companies = list_companies(df)
years = list_years(df)

# -------------------------- LAYOUT: Mode Selector ----------------
mode = st.radio("Choose mode:", ["Chat (Q&A)", "Dashboard"], index=0, horizontal=True)

# -------------------------- CHAT MODE ---------------------------
if mode == "Chat (Q&A)":
    st.subheader("Ask questions in natural language")
    st.write("Examples: \n - What was Tesla's revenue in 2023?\n - Compare Microsoft and Apple revenue in 2022.\n - What is Tesla's liabilities-to-assets ratio in 2023?\n - Show last 3 years revenue trend for Microsoft.")

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Ask a question:", key="chat_input")
    col1, col2 = st.columns([1, 1])
    with col1:
        ask = st.button("🔎 Ask")
    with col2:
        clear = st.button("🧹 Clear")

    if clear:
        st.session_state.history = []

    for role, text, chart_bytes in st.session_state.history:
        if role == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)
            if chart_bytes is not None:
                st.image(chart_bytes)

    if ask and question and question.strip():
        st.session_state.history.append(("user", question, None))
        q_low = question.lower()

        # detect companies, years, metric
        comps = detect_companies_in_text(question, companies)
        yrs = parse_years_in_text(question, years)
        metric = detect_metric(question)

        # If metric is ratio keyword, we will handle separately
        answer_text = ""
        chart_img_bytes = None

        # Multi-company compare
        if is_compare_query(question) and len(comps) >= 2 and yrs:
            y = yrs[0]
            values = []
            for c in comps:
                row = df[(df["Company"] == c) & (df["Fiscal Year"] == y)]
                if not row.empty and metric and metric != "ratio":
                    values.append((c, float(row.iloc[0][metric])))
            if not values:
                answer_text = f"No data available for the requested companies/metric in {y}."
            else:
                # produce bar chart
                comps_plot, vals = zip(*values)
                fig, ax = plt.subplots()
                ax.bar(comps_plot, vals)
                ax.set_title(f"{metric} comparison in {y}")
                ax.set_ylabel(metric)
                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png")
                buf.seek(0)
                chart_img_bytes = buf.getvalue()
                val_txt = ", ".join([f"{c}: {format_billion(v)}" for c, v in values])
                answer_text = f"In {y}, {metric} → {val_txt}."

        # Multi-year trend for one or more companies
        elif ("trend" in q_low or ("last" in q_low and "year" in q_low)) and comps:
            target_years = yrs if yrs else years[-5:]
            fig, ax = plt.subplots()
            plotted = False
            for c in comps:
                subset = df[df["Company"] == c]
                subset = subset[subset["Fiscal Year"].isin(target_years)].sort_values(by="Fiscal Year")
                if metric and not subset.empty:
                    ax.plot(subset["Fiscal Year"], subset[metric], marker="o", label=c)
                    plotted = True
            if not plotted:
                answer_text = "No data to plot for the requested companies/metric/years."
            else:
                ax.set_title(f"{metric} Trend")
                ax.set_ylabel(metric)
                ax.legend()
                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png")
                buf.seek(0)
                chart_img_bytes = buf.getvalue()
                answer_text = f"Showing trend for {', '.join(comps)} over years: {', '.join(map(str, target_years))}."

        # Growth for single company between two years
        elif is_growth_query(question) and len(comps) == 1 and len(yrs) == 2 and metric and metric != "ratio":
            c = comps[0]
            y1, y2 = sorted(yrs)
            row1 = df[(df["Company"] == c) & (df["Fiscal Year"] == y1)]
            row2 = df[(df["Company"] == c) & (df["Fiscal Year"] == y2)]
            if row1.empty or row2.empty:
                answer_text = f"No data for {c} in {y1} or {y2}."
            else:
                v1, v2 = float(row1.iloc[0][metric]), float(row2.iloc[0][metric])
                growth = (v2 - v1) / v1 * 100 if v1 != 0 else None
                if growth is None or math.isinf(growth):
                    answer_text = "Growth cannot be calculated (division by zero)."
                else:
                    fig, ax = plt.subplots()
                    ax.bar([y1, y2], [v1, v2])
                    ax.set_title(f"{c} {metric} Growth")
                    ax.set_ylabel(metric)
                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    chart_img_bytes = buf.getvalue()
                    answer_text = f"{c}'s {metric} grew from {format_billion(v1)} in {y1} to {format_billion(v2)} in {y2} → {growth:.2f}% change."

        # Ratio queries (single company)
        elif "ratio" in q_low and len(comps) == 1:
            c = comps[0]
            target_year = yrs[0] if yrs else max(years)
            row = df[(df["Company"] == c) & (df["Fiscal Year"] == target_year)]
            if row.empty:
                answer_text = f"No data for {c} in {target_year}."
            else:
                liabilities = row.iloc[0].get("Total Liabilities (Billion $)")
                assets = row.iloc[0].get("Total Assets (Billion $)")
                ratio = safe_div(liabilities, assets)
                if ratio is None:
                    answer_text = "Ratio cannot be calculated."
                else:
                    answer_text = f"{c}'s liabilities-to-assets ratio in {target_year} was {ratio:.2f}."

        # Average across years for a company
        elif "average" in q_low and len(comps) == 1 and metric and metric != "ratio":
            c = comps[0]
            subset = df[df["Company"] == c]
            avg = subset[metric].mean()
            if pd.isna(avg):
                answer_text = f"No numeric data to compute average for {c}."
            else:
                answer_text = f"Average {metric} for {c} across available years is {format_billion(avg)}."

        # Direct lookup
        elif len(comps) == 1 and yrs and metric and metric != "ratio":
            c = comps[0]
            y = yrs[0]
            row = df[(df["Company"] == c) & (df["Fiscal Year"] == y)]
            if row.empty:
                answer_text = f"No data for {c} in {y}."
            else:
                val = row.iloc[0][metric]
                answer_text = f"{c} — {metric} in {y}: {format_billion(val)}."

        else:
            answer_text = "I couldn't parse the question fully. Try specifying company name(s), year(s), and metric (e.g., 'Compare Microsoft and Apple revenue in 2022')."

        st.session_state.history.append(("assistant", answer_text, chart_img_bytes))
        st.chat_message("assistant").write(answer_text)
        if chart_img_bytes is not None:
            st.image(chart_img_bytes)

        # Provide option to download result text or chart
        with st.expander("Export results"):
            st.download_button("Download answer as text", answer_text, file_name="answer.txt")
            if chart_img_bytes is not None:
                st.download_button("Download chart PNG", chart_img_bytes, file_name="chart.png", mime="image/png")

# -------------------------- DASHBOARD MODE ---------------------
else:
    st.subheader("Visual Dashboard")
    st.write("Interactively explore companies, metrics and date ranges.")

    left, right = st.columns([1, 2])
    with left:
        sel_companies = st.multiselect("Select company(s)", options=companies, default=companies[:2])
        sel_metric_key = st.selectbox("Select metric", options=list(METRIC_MAP.keys()), index=0)
        sel_metric = METRIC_MAP[sel_metric_key]
        yr_min, yr_max = min(years), max(years)
        sel_years = st.slider("Select year range", min_value=yr_min, max_value=yr_max, value=(max(yr_min, yr_max - 4), yr_max))
        normalize = st.checkbox("Normalize per company (index to first year)", value=False)
        download_csv_button = st.button("Download filtered CSV")

    with right:
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = False
        for c in sel_companies:
            subset = df[(df["Company"] == c) & (df["Fiscal Year"] >= sel_years[0]) & (df["Fiscal Year"] <= sel_years[1])]
            subset = subset.sort_values(by="Fiscal Year")
            if subset.empty:
                continue
            ys = subset["Fiscal Year"].astype(int)
            vals = subset[sel_metric]
            if normalize:
                base = vals.iloc[0]
                if base and base != 0:
                    vals = vals / base * 100
            ax.plot(ys, vals, marker="o", label=c)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No data to display", horizontalalignment='center')
        else:
            ax.set_title(f"{sel_metric} — {sel_years[0]} to {sel_years[1]}")
            ax.set_xlabel("Fiscal Year")
            ax.set_ylabel(sel_metric + (" (Indexed)" if normalize else ""))
            ax.legend()
        st.pyplot(fig)

        # Allow chart download
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("Download chart PNG", buf.getvalue(), file_name="dashboard_chart.png", mime="image/png")

    # filtered CSV download
    if download_csv_button:
        filtered = df[(df["Company"].isin(sel_companies)) & (df["Fiscal Year"] >= sel_years[0]) & (df["Fiscal Year"] <= sel_years[1])]
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv_bytes, file_name="filtered_financials.csv", mime="text/csv")

# -------------------------- FOOTER -----------------------------
st.markdown("---")
st.markdown("**Notes:** Make sure numeric columns are present and named as expected (e.g., 'Total Revenue (Billion $)'). The dashboard and chat rely on those column names. If your CSV uses different column names, rename them in a copy before uploading.")
