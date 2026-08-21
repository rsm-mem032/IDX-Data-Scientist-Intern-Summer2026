from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SIMPLE_MODEL_PATH = OUTPUT_DIR / "streamlit_price_model.joblib"
ADVANCED_MODEL_PATH = OUTPUT_DIR / "streamlit_xgb_county_model.joblib"
SIMPLE_METRICS_PATH = OUTPUT_DIR / "streamlit_app_model_results.csv"
ADVANCED_METRICS_PATH = OUTPUT_DIR / "streamlit_app_advanced_model_results.csv"
PROFILE_PATH = OUTPUT_DIR / "streamlit_app_training_profile.csv"
COUNTY_OPTIONS_PATH = OUTPUT_DIR / "streamlit_county_options.csv"

SIMPLE_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
]
ADVANCED_FEATURES = SIMPLE_FEATURES + ["CountyOrParish"]


@st.cache_resource
def load_joblib_model(path):
    if not path.exists():
        st.error(
            f"Model file not found: {path.name}. "
            "Please rerun notebooks/07_streamlit_app_model.ipynb first."
        )
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def format_currency(value):
    return f"${max(value, 0):,.0f}"


def get_best_metric_row(metrics_df):
    if metrics_df.empty:
        return {
            "model_name": "saved model",
            "R2": float("nan"),
            "MAPE": 0.45,
            "MdAPE": 0.30,
        }
    return metrics_df.sort_values(["MdAPE", "MAPE"]).iloc[0].to_dict()


def show_prediction_block(prediction, metric_row):
    mean_pct_error = float(metric_row["MAPE"])
    median_pct_error = float(metric_row["MdAPE"])
    range_pct = max(mean_pct_error, median_pct_error)

    st.metric("Estimated Close Price", format_currency(prediction))

    col1, col2 = st.columns(2)
    col1.metric("Low Rough Estimate", format_currency(prediction * (1 - range_pct)))
    col2.metric("High Rough Estimate", format_currency(prediction * (1 + range_pct)))

    st.caption(
        "The rough range uses the model's average percentage error from the "
        "May 2026 test set."
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("R-squared", f"{float(metric_row['R2']):.3f}")
    metric_col2.metric("MAPE", f"{mean_pct_error:.1%}")
    metric_col3.metric("Median APE", f"{median_pct_error:.1%}")


def build_input_frame(living_area, bedrooms, bathrooms, lot_size, county=None):
    row = {
        "LivingArea": living_area,
        "BedroomsTotal": bedrooms,
        "BathroomsTotalInteger": bathrooms,
        "LotSizeSquareFeet": lot_size,
    }
    if county is not None:
        row["CountyOrParish"] = county
    return pd.DataFrame([row])


def show_training_comparison(input_df, profile_df):
    if profile_df.empty:
        return

    comparison_rows = []
    for feature in SIMPLE_FEATURES:
        if feature in profile_df["feature"].values:
            row = profile_df.loc[profile_df["feature"] == feature].iloc[0]
            comparison_rows.append(
                {
                    "Feature": feature,
                    "Your Input": float(input_df.loc[0, feature]),
                    "Training Median": float(row["median"]),
                }
            )

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        st.bar_chart(comparison_df.set_index("Feature")[["Your Input", "Training Median"]])


st.set_page_config(page_title="Home Price Prediction Demo")

st.title("Home Price Prediction Demo")
st.write(
    "This Streamlit app shows two Week 9 prediction demos: a required simple "
    "four-input model and an improved county-enhanced XGBoost model."
)

simple_model = load_joblib_model(SIMPLE_MODEL_PATH)
simple_metrics_df = load_csv(SIMPLE_METRICS_PATH)
profile_df = load_csv(PROFILE_PATH)

simple_tab, advanced_tab = st.tabs(["Simple 4-Input Demo", "Advanced County XGBoost"])

with simple_tab:
    st.subheader("Simple 4-Input Demo")
    st.write(
        "This version follows the original Week 9 requirement. It only uses "
        "living area, beds, baths, and lot size. Because it does not include "
        "location, it should be treated as a rough size-based estimate."
    )

    simple_col1, simple_col2 = st.columns(2)
    with simple_col1:
        simple_living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=20000,
            value=1800,
            step=50,
            key="simple_living_area",
        )
        simple_bedrooms = st.number_input(
            "Bedrooms",
            min_value=0.0,
            max_value=12.0,
            value=3.0,
            step=1.0,
            key="simple_bedrooms",
        )
    with simple_col2:
        simple_bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=12.0,
            value=2.0,
            step=0.5,
            key="simple_bathrooms",
        )
        simple_lot_size = st.number_input(
            "Lot Size (sq ft)",
            min_value=500,
            max_value=200000,
            value=6000,
            step=250,
            key="simple_lot_size",
        )

    simple_input_df = build_input_frame(
        simple_living_area, simple_bedrooms, simple_bathrooms, simple_lot_size
    )
    simple_prediction = float(simple_model.predict(simple_input_df[SIMPLE_FEATURES])[0])
    simple_metric_row = get_best_metric_row(simple_metrics_df)

    show_prediction_block(simple_prediction, simple_metric_row)
    st.dataframe(
        simple_input_df.rename(
            columns={
                "BedroomsTotal": "Beds",
                "BathroomsTotalInteger": "Baths",
            }
        ),
        use_container_width=True,
    )
    st.subheader("Input Compared With Training Median")
    show_training_comparison(simple_input_df, profile_df)

with advanced_tab:
    st.subheader("Advanced County XGBoost")
    st.write(
        "This version adds one location input, County, and uses an XGBoost model "
        "with a log-transformed target. It is still simpler than the full Week 7 "
        "model, but it is more realistic than the four-input demo."
    )

    advanced_model = load_joblib_model(ADVANCED_MODEL_PATH)
    advanced_metrics_df = load_csv(ADVANCED_METRICS_PATH)
    county_options_df = load_csv(COUNTY_OPTIONS_PATH)
    county_options = (
        county_options_df["CountyOrParish"].dropna().sort_values().tolist()
        if not county_options_df.empty
        else ["Los Angeles", "Orange", "Riverside", "San Bernardino", "San Diego"]
    )

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        advanced_county = st.selectbox(
            "County", county_options, index=county_options.index("Los Angeles")
            if "Los Angeles" in county_options
            else 0
        )
        advanced_living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=20000,
            value=1800,
            step=50,
            key="advanced_living_area",
        )
        advanced_bedrooms = st.number_input(
            "Bedrooms",
            min_value=0.0,
            max_value=12.0,
            value=3.0,
            step=1.0,
            key="advanced_bedrooms",
        )
    with adv_col2:
        advanced_bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=12.0,
            value=2.0,
            step=0.5,
            key="advanced_bathrooms",
        )
        advanced_lot_size = st.number_input(
            "Lot Size (sq ft)",
            min_value=500,
            max_value=200000,
            value=6000,
            step=250,
            key="advanced_lot_size",
        )

    advanced_input_df = build_input_frame(
        advanced_living_area,
        advanced_bedrooms,
        advanced_bathrooms,
        advanced_lot_size,
        county=advanced_county,
    )
    advanced_prediction = float(advanced_model.predict(advanced_input_df[ADVANCED_FEATURES])[0])
    advanced_metric_row = get_best_metric_row(advanced_metrics_df)

    show_prediction_block(advanced_prediction, advanced_metric_row)
    st.dataframe(
        advanced_input_df.rename(
            columns={
                "BedroomsTotal": "Beds",
                "BathroomsTotalInteger": "Baths",
            }
        ),
        use_container_width=True,
    )
    st.subheader("Input Compared With Training Median")
    show_training_comparison(advanced_input_df, profile_df)

    st.info(
        "This advanced app still does not include list price, year built, latitude, "
        "longitude, school district, or market-context features. The full Week 7 "
        "XGBoost model remains the stronger modeling approach."
    )

