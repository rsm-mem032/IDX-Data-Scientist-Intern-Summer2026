from html import escape
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SALES_PATH = OUTPUTS / "train_preprocessed.csv"
COUNTIES_PATH = OUTPUTS / "streamlit_county_options.csv"
ADVANCED_MODEL_PATH = OUTPUTS / "streamlit_xgb_county_model.joblib"
SIMPLE_MODEL_PATH = OUTPUTS / "streamlit_price_model.joblib"
ADVANCED_METRICS_PATH = OUTPUTS / "streamlit_app_advanced_model_results.csv"
SIMPLE_METRICS_PATH = OUTPUTS / "streamlit_app_model_results.csv"
ADVANCED_PREDICTIONS_PATH = OUTPUTS / "streamlit_app_advanced_predictions.csv"

SIMPLE_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
]
ADVANCED_FEATURES = SIMPLE_FEATURES + ["CountyOrParish"]

st.set_page_config(
    page_title="California Home Price Studio",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    :root { --ink:#173b3a; --muted:#586b68; --accent:#bf593b; --paper:#f6f4ee; --line:#d9e2dc; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans', sans-serif; }
    .block-container { max-width:1160px; padding-top:2.4rem; padding-bottom:4rem; }
    [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 { font-family:'Fraunces', Georgia, serif !important; color:var(--ink); letter-spacing:-.035em; }
    h1 { font-size:clamp(2.8rem,5vw,4.7rem) !important; line-height:1.07; }
    h2 { font-size:clamp(2rem,3vw,2.8rem); }
    p, label { font-family:'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    #MainMenu, footer { visibility:hidden; }
    .brand-row { display:flex; justify-content:space-between; align-items:center; gap:1rem; padding-bottom:1.15rem; border-bottom:1px solid var(--line); }
    .brand-name { font-family:'Fraunces', Georgia, serif; font-size:1.5rem; font-weight:600; color:var(--ink); }
    .brand-tag { color:var(--muted); font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; }
    .eyebrow { color:var(--accent); font-size:.76rem; font-weight:700; letter-spacing:.17em; text-transform:uppercase; margin-bottom:.8rem; }
    .hero { padding:3rem 0 1.6rem; }
    .hero h1 { margin:0 0 1.1rem; max-width:860px; }
    .hero p { color:var(--muted); font-size:1.07rem; line-height:1.7; max-width:720px; margin:0; }
    .detail-strip { border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:.9rem 0; color:var(--muted); font-size:.82rem; margin:1.2rem 0 2.2rem; }
    .section-kicker { color:var(--accent); font-size:.75rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin:1.2rem 0 .5rem; }
    .section-copy { color:var(--muted); line-height:1.65; max-width:780px; }
    .quiet-card { border:1px solid var(--line); border-radius:16px; padding:1.6rem; background:#fff; min-height:180px; }
    .quiet-card h3 { margin:.4rem 0 .8rem; font-size:1.45rem; }
    .quiet-card p { color:var(--muted); line-height:1.65; margin:0; }
    .result-card { background:var(--ink); color:#fff; border-radius:18px; padding:2rem 2.2rem; margin:1rem 0; }
    .result-card .eyebrow { color:#e6b6a6; }
    .result-card .price { font-family:'Fraunces', Georgia, serif; font-size:clamp(2.8rem,5vw,4.5rem); line-height:1.05; margin:.45rem 0; }
    .result-card .result-note { color:#c8d8d3; font-size:.9rem; line-height:1.6; }
    .mini-label { color:var(--muted); font-size:.75rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .big-number { font-family:'Fraunces', Georgia, serif; font-size:1.75rem; color:var(--ink); margin:.3rem 0; }
    .summary-card { border:1px solid var(--line); border-radius:14px; background:#fff; padding:1.15rem 1.3rem; min-height:126px; }
    .summary-card .value { font-family:'Fraunces', Georgia, serif; font-size:clamp(1.5rem,2vw,2rem); line-height:1.2; color:var(--ink); margin:.45rem 0 0; }
    .summary-card .helper { color:var(--muted); font-size:.74rem; margin-top:.4rem; }
    .footnote { color:var(--muted); font-size:.82rem; line-height:1.6; }
    [data-testid="stForm"] { border:1px solid var(--line); border-radius:18px; background:#fff; padding:1.3rem 1.5rem; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.2rem; }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background:var(--accent); border-color:var(--accent); border-radius:9px; color:white; }
    div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover { background:#a7492f; border-color:#a7492f; }
    .stRadio [role="radiogroup"] { gap:.35rem; }
    .stRadio [role="radiogroup"] label { border:1px solid var(--line); border-radius:100px; padding:.2rem .65rem; background:#fff; }
    .site-footer { border-top:1px solid var(--line); padding-top:1.3rem; margin-top:3.4rem; color:var(--muted); font-size:.82rem; }
    @media (max-width:640px) { .block-container { padding-top:1.2rem; } .brand-tag { display:none; } .hero { padding-top:2rem; } .result-card { padding:1.5rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run notebooks/07_streamlit_app_model.ipynb first."
        )
    return joblib.load(path)


@st.cache_data
def load_sales():
    columns = [
        "ClosePrice", "close_month", "CountyOrParish", "LivingArea",
        "BedroomsTotal", "LotSizeSquareFeet",
    ]
    sales = pd.read_csv(SALES_PATH, usecols=columns, parse_dates=["close_month"])
    sales = sales.dropna(subset=["ClosePrice", "CountyOrParish", "LivingArea"])
    sales = sales.loc[(sales["ClosePrice"] > 0) & (sales["LivingArea"] > 0)].copy()
    sales["price_per_sqft"] = sales["ClosePrice"] / sales["LivingArea"]
    return sales


@st.cache_data
def load_counties():
    counties = pd.read_csv(COUNTIES_PATH)["CountyOrParish"].dropna().unique()
    return sorted(str(county) for county in counties)


@st.cache_data
def load_metrics(path, model_name):
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    match = rows.loc[rows["model_name"] == model_name]
    return match.iloc[0].to_dict() if not match.empty else None


@st.cache_data
def load_segment_errors():
    if not ADVANCED_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(
        ADVANCED_PREDICTIONS_PATH, usecols=["ClosePrice", "ape"]
    ).dropna()
    predictions["price_band"] = pd.cut(
        predictions["ClosePrice"],
        bins=[0, 500_000, 1_000_000, 2_000_000, 5_000_000, np.inf],
        labels=["Under $500k", "$500k-$1m", "$1m-$2m", "$2m-$5m", "$5m+"],
    )
    summary = predictions.groupby("price_band", observed=True).agg(
        sales=("ape", "size"), typical_error=("ape", "median")
    ).reset_index()
    summary = summary.loc[summary["sales"] >= 30].copy()
    summary["typical_error"] *= 100
    return summary


def money(value):
    return f"${value:,.0f}"


def short_money(value):
    return f"${value / 1_000_000:.2f}m" if value >= 1_000_000 else f"${value / 1_000:.0f}k"


def input_frame(area, beds, baths, lot, county):
    return pd.DataFrame([{
        "LivingArea": area,
        "BedroomsTotal": beds,
        "BathroomsTotalInteger": baths,
        "LotSizeSquareFeet": lot,
        "CountyOrParish": county,
    }])


def comparable_sales(sales, county, area, beds):
    local = sales.loc[sales["CountyOrParish"] == county]
    recent_start = sales["close_month"].max() - pd.DateOffset(months=5)
    recent = local.loc[local["close_month"] >= recent_start]
    for tolerance in (0.25, 0.40):
        matches = recent.loc[
            recent["LivingArea"].between(area * (1 - tolerance), area * (1 + tolerance))
            & recent["BedroomsTotal"].between(max(0, beds - 1), beds + 1)
        ]
        if len(matches) >= 20:
            return matches, (
                f"{county} sales from the latest six training months, within "
                f"{tolerance:.0%} of this living area and within one bedroom"
            )
    older_matches = local.loc[
        local["LivingArea"].between(area * 0.60, area * 1.40)
        & local["BedroomsTotal"].between(max(0, beds - 1), beds + 1)
    ]
    if len(older_matches) >= 20:
        return older_matches, (
            f"{county} sales across the full training period, within 40% of this "
            "living area and within one bedroom; recent matches were limited"
        )
    if len(recent) >= 20:
        return recent, f"All recent {county} sales; too few similar homes for a narrower comparison"
    return local, f"All recorded {county} homes; too few similar homes for a narrower comparison"


def note_card(title, body):
    st.markdown(
        f'<div class="quiet-card"><div class="eyebrow">Decision guide</div>'
        f'<h3>{escape(title)}</h3><p>{escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def summary_card(label, value, helper):
    st.markdown(
        f'<div class="summary-card"><div class="mini-label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        f'<div class="helper">{escape(helper)}</div></div>',
        unsafe_allow_html=True,
    )


def show_comparable_chart(comps, prediction):
    low, high = comps["ClosePrice"].quantile([0.01, 0.99])
    chart_data = comps.loc[comps["ClosePrice"].between(low, high)]
    bars = alt.Chart(chart_data).mark_bar(color="#91b4a8", cornerRadiusTopLeft=3).encode(
        x=alt.X("ClosePrice:Q", bin=alt.Bin(maxbins=24), title="Historical close price", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("count():Q", title="Homes sold"),
        tooltip=[alt.Tooltip("count():Q", title="Homes")],
    )
    if low <= prediction <= high:
        marker = alt.Chart(pd.DataFrame({"estimate": [prediction]})).mark_rule(
            color="#bf593b", strokeWidth=3
        ).encode(x="estimate:Q", tooltip=[alt.Tooltip("estimate:Q", format="$,.0f")])
        bars = bars + marker
    st.altair_chart(bars.properties(height=265), width="stretch")
    st.caption(
        "Historical transactions are trimmed to the 1st-99th percentile for chart readability. "
        "The orange line marks this model estimate when it falls within the chart range."
    )


st.markdown(
    '<div class="brand-row"><span class="brand-name">California Home Price Studio</span>'
    '<span class="brand-tag">An IDX data science project / 2026</span></div>',
    unsafe_allow_html=True,
)
page = st.radio(
    "Explore",
    ["Estimate a home", "Explore the market", "Model & limitations"],
    horizontal=True,
    label_visibility="collapsed",
)

try:
    sales = load_sales()
    counties = load_counties()
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error(f"Required project data could not be loaded: {exc}")
    st.stop()

history_start = sales["close_month"].min().strftime("%B %Y")
history_end = sales["close_month"].max().strftime("%B %Y")

if page == "Estimate a home":
    st.markdown(
        '<div class="hero"><div class="eyebrow">Property intelligence / California</div>'
        '<h1>Price a California home.<br>See the local picture.</h1>'
        '<p>Enter five details to see a predicted close price, then compare it with '
        'historical sales of similar homes in the same county.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="detail-strip">Single-family residences &nbsp; / &nbsp; '
        'County-aware XGBoost &nbsp; / &nbsp; May 2026 evaluation</div>',
        unsafe_allow_html=True,
    )

    form_col, guide_col = st.columns([1.75, 1], gap="large")
    with form_col:
        st.markdown('<div class="section-kicker">01 / Tell us about the home</div>', unsafe_allow_html=True)
        with st.form("estimate_form"):
            county = st.selectbox(
                "County", counties,
                index=counties.index("Los Angeles") if "Los Angeles" in counties else 0,
            )
            left, right = st.columns(2, gap="medium")
            with left:
                area = st.number_input("Living area (sq ft)", 300, 20_000, 1_800, 50)
                beds = st.number_input("Bedrooms", 0, 12, 3, 1)
            with right:
                baths = st.number_input("Bathrooms", 0.0, 12.0, 2.0, 0.5)
                lot = st.number_input("Lot size (sq ft)", 0, 200_000, 6_000, 250)
            submitted = st.form_submit_button(
                "Calculate estimate", type="primary", width="stretch"
            )
        st.caption("Prices reflect historical closed sales, not current listings or a formal appraisal.")

    with guide_col:
        st.markdown('<div class="section-kicker">How to use this number</div>', unsafe_allow_html=True)
        note_card(
            "A starting point for a pricing conversation",
            "Compare the estimate with recent local sales. Condition, renovations, views, "
            "and exact street location still need human review.",
        )
        st.markdown(
            '<p class="footnote">The model uses county and four property measurements. '
            'It does not know the address or interior condition.</p>',
            unsafe_allow_html=True,
        )

    if submitted:
        features = input_frame(area, beds, baths, lot, county)
        try:
            estimate = float(load_model(ADVANCED_MODEL_PATH).predict(features[ADVANCED_FEATURES])[0])
        except (FileNotFoundError, ValueError, ImportError) as exc:
            st.error(f"The saved prediction model could not be used: {exc}")
        else:
            if not np.isfinite(estimate) or estimate <= 0:
                st.error("The model returned an invalid estimate for these inputs.")
            else:
                st.session_state["estimate_result"] = {
                    "price": estimate, "area": area, "beds": beds,
                    "baths": baths, "lot": lot, "county": county,
                }

    result = st.session_state.get("estimate_result")
    if result:
        estimate = result["price"]
        result_county = result["county"]
        result_area = result["area"]
        result_beds = result["beds"]
        st.markdown('<div class="section-kicker">02 / Your estimate</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-card"><div class="eyebrow">Estimated close price</div>'
            f'<div class="price">{money(estimate)}</div>'
            f'<div class="result-note">{escape(result_county)} County &nbsp; / &nbsp; '
            f'{result_area:,} sq ft &nbsp; / &nbsp; {money(estimate / result_area)} per sq ft<br>'
            'Generated by the saved five-input County XGBoost model.</div></div>',
            unsafe_allow_html=True,
        )

        area_upper = sales["LivingArea"].quantile(0.99)
        lot_upper = sales["LotSizeSquareFeet"].quantile(0.99)
        if result_area > area_upper or result["lot"] > lot_upper:
            st.warning(
                "At least one size input is above the 99th percentile of the training homes. "
                "The estimate may be less reliable for this unusual property."
            )

        comps, comparison_label = comparable_sales(
            sales, result_county, result_area, result_beds
        )
        st.markdown('<div class="section-kicker">03 / Local sales perspective</div>', unsafe_allow_html=True)
        st.subheader("How does this compare with closed sales?")
        st.write(comparison_label)
        if len(comps) >= 20:
            lower = comps["ClosePrice"].quantile(0.25)
            upper = comps["ClosePrice"].quantile(0.75)
            c1, c2, c3 = st.columns(3)
            with c1:
                summary_card("Historical median", money(comps["ClosePrice"].median()), "Recorded close prices")
            with c2:
                summary_card("Middle 50% of sales", f"{short_money(lower)} - {short_money(upper)}", "25th to 75th percentile")
            with c3:
                summary_card("Sales in comparison", f"{len(comps):,}", "Matching historical homes")
            show_comparable_chart(comps, estimate)
            lower_label = money(lower).replace("$", r"\$")
            upper_label = money(upper).replace("$", r"\$")
            st.caption(
                f"Source: training-period closed sales, {history_start} to {history_end}. "
                f"The middle 50% spans {lower_label} to {upper_label}. "
                "This is a broad historical reference, not a matched appraisal or a prediction interval."
            )
        else:
            st.info("There are too few historical sales in this county for a useful comparison.")

        with st.expander("Compare with the original four-input coursework model"):
            st.write(
                "This model ignores county. It fulfills the original four-input app task, "
                "but has weaker May 2026 test performance."
            )
            simple_features = input_frame(
                result_area, result_beds, result["baths"], result["lot"], result_county
            )
            try:
                simple_estimate = float(
                    load_model(SIMPLE_MODEL_PATH).predict(simple_features[SIMPLE_FEATURES])[0]
                )
                st.metric("Four-input model estimate", money(simple_estimate))
            except (FileNotFoundError, ValueError, ImportError) as exc:
                st.warning(f"The comparison model is unavailable: {exc}")
            st.caption("Two model estimates are not a confidence range.")

elif page == "Explore the market":
    st.markdown(
        '<div class="hero"><div class="eyebrow">Historical market context</div>'
        '<h1>Every county has its own price story.</h1>'
        '<p>Explore closed-sale patterns behind the model. These charts describe past '
        'transactions and are not a live housing-market feed.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="detail-strip">Source: {escape(history_start)} to '
        f'{escape(history_end)} training-period closed sales &nbsp; / &nbsp; '
        f'{len(sales):,} usable records</div>',
        unsafe_allow_html=True,
    )
    market_county = st.selectbox(
        "Choose a county", counties,
        index=counties.index("Los Angeles") if "Los Angeles" in counties else 0,
        key="market_county",
    )
    local = sales.loc[sales["CountyOrParish"] == market_county]
    if local.empty:
        st.info("No training-period sales are available for this county.")
    else:
        a, b, c = st.columns(3)
        a.metric("Recorded sales", f"{len(local):,}")
        b.metric("Median close price", money(local["ClosePrice"].median()))
        c.metric("Median price per sq ft", money(local["price_per_sqft"].median()))

        monthly = local.groupby("close_month", as_index=False).agg(
            median_price=("ClosePrice", "median"), sales=("ClosePrice", "size")
        )
        st.markdown('<div class="section-kicker">Price over time</div>', unsafe_allow_html=True)
        st.subheader(f"{market_county} monthly median close price")
        price_chart = alt.Chart(monthly).mark_line(
            color="#bf593b", point=alt.OverlayMarkDef(color="#bf593b"), strokeWidth=3
        ).encode(
            x=alt.X("close_month:T", title="Close month"),
            y=alt.Y("median_price:Q", title="Median close price", axis=alt.Axis(format="$,.0f"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("close_month:T", title="Month", format="%B %Y"),
                     alt.Tooltip("median_price:Q", title="Median", format="$,.0f"),
                     alt.Tooltip("sales:Q", title="Sales")],
        ).properties(height=320)
        st.altair_chart(price_chart, width="stretch")
        st.caption(
            "The vertical axis is narrowed to show monthly movement. "
            "Monthly medians can shift when the mix of homes sold changes. "
            "This line alone does not measure appreciation of the same home."
        )

        st.markdown('<div class="section-kicker">Market activity</div>', unsafe_allow_html=True)
        st.subheader("Closed sales by month")
        volume_chart = alt.Chart(monthly).mark_bar(
            color="#8cae9d", cornerRadiusTopLeft=3
        ).encode(
            x=alt.X("close_month:T", title="Close month"),
            y=alt.Y("sales:Q", title="Homes sold"),
            tooltip=[alt.Tooltip("close_month:T", title="Month", format="%B %Y"),
                     alt.Tooltip("sales:Q", title="Sales")],
        ).properties(height=220)
        st.altair_chart(volume_chart, width="stretch")
        st.caption("Counts include only homes retained in the project's preprocessing pipeline.")

else:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Method & transparency</div>'
        '<h1>Know what the estimate can tell you.</h1>'
        '<p>The live estimate uses the saved five-input County XGBoost model. '
        'Its May 2026 test results below belong to that same model.</p></div>',
        unsafe_allow_html=True,
    )
    advanced_metrics = load_metrics(
        ADVANCED_METRICS_PATH, "xgb_log_county_5_inputs"
    )
    simple_metrics = load_metrics(
        SIMPLE_METRICS_PATH, "hist_gradient_boosting_log_4_inputs"
    )
    st.markdown('<div class="section-kicker">Historical evaluation / May 2026</div>', unsafe_allow_html=True)
    st.subheader("How the online model performed")
    if advanced_metrics:
        a, b, c = st.columns(3)
        a.metric("R-squared", f"{advanced_metrics['R2']:.3f}")
        b.metric("Typical error / MdAPE", f"{advanced_metrics['MdAPE']:.1%}")
        c.metric("Average error / MAPE", f"{advanced_metrics['MAPE']:.1%}")
        st.write(
            "For the middle test home, the absolute percentage miss was about "
            f"{advanced_metrics['MdAPE']:.0%}. Some homes had much larger errors, "
            "so this is a pricing screen, not a guaranteed valuation."
        )
        st.caption(
            "May 2026 was also used to compare app model candidates. These reported "
            "scores may be optimistic; a later untouched month would be a stronger final check."
        )
    else:
        st.warning("Saved May 2026 evaluation metrics are unavailable.")

    st.markdown('<div class="section-kicker">Performance varies by price</div>', unsafe_allow_html=True)
    segments = load_segment_errors()
    if not segments.empty:
        segment_chart = alt.Chart(segments).mark_bar(
            color="#3b7774", cornerRadiusEnd=4
        ).encode(
            x=alt.X("typical_error:Q", title="Median absolute percentage error (%)"),
            y=alt.Y("price_band:N", title=None, sort=None),
            tooltip=[alt.Tooltip("price_band:N", title="Actual sale price band"),
                     alt.Tooltip("typical_error:Q", title="Typical error", format=".1f"),
                     alt.Tooltip("sales:Q", title="Test homes")],
        ).properties(height=245)
        st.altair_chart(segment_chart, width="stretch")
        st.caption(
            "Bands use actual sale prices from the May 2026 test set. "
            "They describe past errors, not the uncertainty of a specific new home."
        )

    left, right = st.columns(2, gap="large")
    with left:
        note_card(
            "What the model sees",
            "Living area, bedrooms, bathrooms, lot size, and county. The saved model "
            "was trained on earlier single-family closed sales and predicts close price.",
        )
    with right:
        note_card(
            "What it cannot see",
            "Street-level location, interior condition, remodel quality, school assignment, "
            "views, and current listing competition. These can change value materially.",
        )

    st.markdown('<div class="section-kicker">Coursework comparison</div>', unsafe_allow_html=True)
    st.subheader("Why there are two saved app models")
    st.write(
        "The original four-input model demonstrates the Week 9 requirement. "
        "Adding county gives the live model a basic location signal, but it still "
        "does not have the richer information used in the earlier research notebooks."
    )
    if advanced_metrics and simple_metrics:
        comparison = pd.DataFrame([
            {"Model": "Live County XGBoost", "Inputs": 5,
             "R2": advanced_metrics["R2"], "MdAPE": advanced_metrics["MdAPE"],
             "MAPE": advanced_metrics["MAPE"]},
            {"Model": "Original four-input baseline", "Inputs": 4,
             "R2": simple_metrics["R2"], "MdAPE": simple_metrics["MdAPE"],
             "MAPE": simple_metrics["MAPE"]},
        ])
        st.dataframe(
            comparison.style.format({"R2": "{:.3f}", "MdAPE": "{:.1%}", "MAPE": "{:.1%}"}),
            hide_index=True, width="stretch",
        )
    st.info(
        "The fuller Week 7 XGBoost experiments report different scores because they "
        "use more features. Their results are not the accuracy of this five-input app."
    )

st.markdown(
    '<div class="site-footer">California Home Price Studio &nbsp; / &nbsp; '
    'Historical CRMLS single-family closed sales &nbsp; / &nbsp; '
    'A data science portfolio project, not an appraisal service. &nbsp; / &nbsp; '
    '<a href="https://github.com/rsm-mem032/IDX-Data-Scientist-Intern-Summer2026" target="_blank">View source code</a></div>',
    unsafe_allow_html=True,
)
