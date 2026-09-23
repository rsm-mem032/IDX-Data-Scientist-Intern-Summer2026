# California Home Price Prediction

An end-to-end machine learning project completed during my 12-week Data Science Internship at IDXExchange. The project uses historical California residential sales data to estimate home closing prices and explore how property and location features affect prediction quality.

## Project Overview

The target variable is `ClosePrice`. The project covers the full workflow from raw data preparation to model evaluation and an interactive Streamlit application:

- Explore and clean monthly CRMLS sold-property records.
- Build a reproducible preprocessing pipeline.
- Compare interpretable and tree-based regression models.
- Engineer real estate features, including county and school district information.
- Evaluate models with a time-based holdout month to reduce leakage.
- Deploy a small Streamlit app for interactive price estimates.

## Dataset

The dataset contains monthly CRMLS sold-property extracts covering June 2025 through May 2026. The modeling scope is residential single-family properties:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

The main modeling split is:

- Training period: June 2025 through April 2026
- Test month: May 2026

The raw files are kept locally in `data/raw/`. They are not included in the repository.

## Modeling Workflow

### Preprocessing

The preprocessing pipeline includes:

- Combining the monthly CRMLS files.
- Filtering to the project property scope.
- Removing duplicate listing records.
- Converting price, size, bed, bath, lot, location, and date fields to usable types.
- Removing invalid target values and clearly invalid property records.
- Creating `close_month` for time-based splitting.
- Creating features such as `property_age`, `log_living_area`, and `bathrooms_per_bedroom`.

### Feature Engineering

The project tested several real estate features:

- Price per square foot.
- Lot-to-living-area ratio.
- Property age and newer-home flags.
- Bedroom and bathroom ratios.
- Log-transformed size and market-time features.
- County information.
- Unified school district mapping using California school district boundary data.

The experiments showed that adding more features did not automatically improve performance. Feature selection, target transformation, and time-based evaluation were as important as model complexity.

### Models Tested

- Linear Regression baseline.
- Decision Tree Regressor.
- Random Forest Regressor.
- XGBoost with a log-transformed target.
- Simple app models using only the four required property inputs.
- County-enhanced XGBoost model used by the Streamlit app.

## Main Model Results

The main research models were evaluated on the May 2026 test month.

| Model | R2 | MAPE | MdAPE | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| XGBoost reality features | 0.6347 | 7.26% | 2.53% | $74,895 | $1,013,075 |
| XGBoost market-context features | 0.6345 | 7.18% | 2.49% | $74,596 | $1,013,250 |
| Linear Regression baseline | 0.6336 | 24.56% | 10.14% | $198,949 | $1,014,591 |

The XGBoost models produced much lower percentage errors than the linear baseline for typical homes. The large RMSE reflects the long right tail of California home prices and the difficulty of predicting unusual or luxury properties.

## Streamlit Application

The app is available in `app.py` and is designed as a decision-support demo rather than a formal appraisal tool.

### Estimate A Home

The main estimate uses a saved five-input County XGBoost model. Users enter:

- County
- Living area
- Bedrooms
- Bathrooms
- Lot size

The app returns a predicted close price and price per square foot. It also compares the estimate with historical closed sales from the same county. The comparison first searches the latest six training months for homes with similar living area and bedroom count. If there are too few matches, it widens the comparison and clearly labels the broader reference.

### Explore The Market

Users can select a county and view:

- Historical median close price by month.
- Closed sales volume by month.
- Median price per square foot for the selected county.

These charts describe historical transactions from June 2025 through April 2026. They are not live market data.

### Model And Limitations

The app reports the evaluation metrics for the same five-input model used by the estimate:

| App Model | R2 | MAPE | MdAPE |
|---|---:|---:|---:|
| County-enhanced XGBoost | 0.3660 | 30.54% | 20.22% |
| Simple four-input model | 0.3042 | 44.00% | 31.20% |

The four-input model is retained to satisfy the original app requirement. The County XGBoost model is used as the main app model because it includes a basic location signal.

May 2026 was also used to compare app model candidates, so these app metrics may be somewhat optimistic. A later untouched month would provide a stronger final production evaluation.

## Repository Structure

```text
IDX-Data-Scientist-Intern-Summer2026/
├── app.py
├── README.md
├── requirements.txt
├── metadata_notes.md
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_streamlit_app_model.ipynb
├── outputs/
└── src/
```

## How To Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

On macOS, install the OpenMP runtime required by XGBoost if needed:

```bash
brew install libomp
```

On the remote project environment, use:

```bash
/opt/base-uv/.venv/bin/python -m pip install -r requirements.txt
```

### Re-run The Notebooks

Place the monthly CRMLS files in `data/raw/`, then run the notebooks in order:

```text
01_exploration.ipynb
02_preprocessing.ipynb
03_baseline_model.ipynb
04_model_comparison.ipynb
05_feature_engineering.ipynb
05_advanced_models.ipynb
06_evaluation.ipynb
07_streamlit_app_model.ipynb
```

The last notebook saves the models and CSV files used by the app in `outputs/`.

### Launch The App

From the repository root:

```bash
streamlit run app.py
```

If the command is unavailable:

```bash
python -m streamlit run app.py
```

## Limitations

- The app does not use exact street location, interior condition, renovation quality, views, or current listing competition.
- The app model is simpler than the full research models because it is limited to the information a user can enter easily.
- A prediction is a screening estimate, not a formal appraisal, lending decision, or offer to buy.
- Unusual, luxury, and thin-market homes require additional comparable-sale and professional review.

## Author

Mia Ma
