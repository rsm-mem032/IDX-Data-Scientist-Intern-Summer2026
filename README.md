<div align="center">

# California Home Price Intelligence

### An end-to-end machine learning project for residential price estimation in California

Built during a 12-week Data Science Internship at **IDXExchange**.

<p>
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-modeling-189FDD?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/Streamlit-app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/CRMLS-sold%20property%20data-173B3A?style=flat-square" alt="CRMLS">
</p>

</div>

## What I Built

This project turns monthly California sold-property records into an interactive price review tool. The workflow covers data preparation, time-based validation, model comparison, real estate feature engineering, and a Streamlit application.

The app lets a user enter a home's county, living area, bedrooms, bathrooms, and lot size. It returns a predicted closing price and places that estimate next to historical sales from the same county.

## Project Snapshot

| Area | Details |
|---|---|
| Business question | How can we estimate a home's closing price while making the result easier to interpret? |
| Target | `ClosePrice` |
| Property scope | California residential single-family homes |
| Data window | June 2025 through May 2026 |
| Evaluation design | Time-based holdout with May 2026 as the test month |
| Main research model | XGBoost with a log-transformed target |
| Interactive output | County-aware Streamlit price estimate |

## Key Results

### Research Models

The main research models were evaluated on the May 2026 test month.

| Model | R2 | MAPE | MdAPE | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| XGBoost with reality features | 0.6347 | 7.26% | 2.53% | $74,895 | $1,013,075 |
| XGBoost with market-context features | 0.6345 | 7.18% | 2.49% | $74,596 | $1,013,250 |
| Linear Regression baseline | 0.6336 | 24.56% | 10.14% | $198,949 | $1,014,591 |

The XGBoost experiments produced much lower percentage errors than the linear baseline for typical homes. RMSE remained high because California home prices have a long right tail and unusual or luxury properties are harder to predict.

### Streamlit App Model

The live app uses a simpler five-input County XGBoost model so that a user can provide the inputs directly.

| App model | Inputs | R2 | MAPE | MdAPE |
|---|---:|---:|---:|---:|
| County-enhanced XGBoost | 5 | 0.3660 | 30.54% | 20.22% |
| Four-input coursework baseline | 4 | 0.3042 | 44.00% | 31.20% |

The online app model is intentionally different from the strongest full research model. It gives up some predictive power in exchange for a simpler user experience and fewer required inputs.

## Product Experience

### 1. Estimate A Home

The primary view accepts:

- County
- Living area
- Bedrooms
- Bathrooms
- Lot size

It returns the estimated close price and price per square foot. The result is followed by a historical comparison using recent sales from the same county and similar home-size and bedroom ranges.

### 2. Explore The Market

The market view provides county-level historical context:

- Monthly median close price.
- Monthly closed-sale volume.
- Median price per square foot.

The charts use the project's historical training-period sales. They are designed to explain the data behind the estimate, not to act as a live market feed.

### 3. Model And Limitations

The transparency view shows the metrics for the exact model used in the app, price-segment error patterns, input features, and the information the model cannot see.

## Modeling Approach

### Preprocessing

- Combine the monthly CRMLS sold-property extracts.
- Filter to residential single-family properties.
- Remove duplicate listing records.
- Convert price, size, bed, bath, lot, location, and date fields to usable types.
- Remove invalid target values and clearly invalid property records.
- Create `close_month` for time-based splitting.
- Create features such as `property_age`, `log_living_area`, and `bathrooms_per_bedroom`.

### Feature Engineering

The project tested:

- Price per square foot.
- Lot-to-living-area ratio.
- Property age and newer-home flags.
- Bedroom and bathroom ratios.
- Log-transformed size and market-time features.
- County information.
- Unified school district mapping using California school district boundary data.

The experiments showed that additional features did not automatically improve performance. Feature selection, target transformation, and evaluation design were as important as model complexity.

### Models Tested

- Linear Regression.
- Decision Tree Regressor.
- Random Forest Regressor.
- XGBoost with a log-transformed target.
- Simple app models using the four required property inputs.
- County-enhanced XGBoost for the interactive app.

## Evaluation Design

The project uses a time-based split to better reflect how a production model would be used:

```text
June 2025 - April 2026  ->  training data
May 2026                ->  evaluation month
```

The main metrics are:

- **R2:** how much variation in closing price the model explains.
- **MAPE:** average absolute percentage error.
- **MdAPE:** median absolute percentage error, which describes the typical case more robustly than MAPE.
- **MAE:** average dollar error.
- **RMSE:** a metric that gives extra weight to large errors.

May 2026 was also used to compare the app model candidates. The app results may therefore be somewhat optimistic. A later untouched month would provide a stronger final production evaluation.

## Repository Structure

```text
IDX-Data-Scientist-Intern-Summer2026/
├── app.py                         # Streamlit application
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── metadata_notes.md              # Data dictionary and notes
├── data/raw/                      # Local CRMLS source files
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_streamlit_app_model.ipynb
├── outputs/                      # Saved models, metrics, and predictions
└── src/                          # Supporting project code
```

## Run The Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

On macOS, install the OpenMP runtime required by XGBoost if needed:

```bash
brew install libomp
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

The final notebook saves the serialized models and CSV files used by the app in `outputs/`.

### Launch The App

From the repository root:

```bash
streamlit run app.py
```

If the `streamlit` command is unavailable:

```bash
python -m streamlit run app.py
```

## Limitations

- The app does not use exact street location, interior condition, renovation quality, views, or current listing competition.
- The app model is simpler than the full research models because it is limited to information a user can enter easily.
- A prediction is a screening estimate, not a formal appraisal, lending decision, or offer to buy.
- Unusual, luxury, and thin-market homes require additional comparable-sale and professional review.
- The raw CRMLS files are not included in this public repository.

## Author

**Mia Ma**

Data Science Intern | Machine Learning | Real Estate Analytics
