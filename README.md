# IDX Data Scientist Intern Summer 2026

## Project Overview

This project predicts California single-family residential home sale prices using historical CRMLS sold property data. The target variable is `ClosePrice`.

The project follows a weekly modeling workflow:

- Explore and clean the raw sold-property data.
- Build a reproducible preprocessing pipeline.
- Train a baseline model.
- Compare tree-based models and advanced XGBoost models.
- Add feature engineering, including school district mapping.
- Evaluate models using the same test month.
- Build a simple Streamlit prediction app.

## Dataset Source

The raw dataset comes from monthly CRMLS sold-property CSV extracts. The modeling window uses closed sales from:

- `2025-06` through `2026-05`

The standardized split is:

- Training set: `2025-06` through `2026-04`
- Test set: `2026-05`

The main project scope is limited to:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

Raw files are stored locally in `data/raw/`. The school district boundary file is also stored locally and used for Week 6 feature engineering.

## Repository Structure

```text
IDX-Data-Scientist-Intern-Summer2026/
├── app.py
├── README.md
├── requirements.txt
├── metadata_notes.md
├── data/
│   ├── raw/
│   └── processed/
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

## Preprocessing Summary

The preprocessing workflow is mainly handled in `notebooks/02_preprocessing.ipynb`.

Key preprocessing steps:

- Load the 12 monthly CRMLS sold files from `2025-06` to `2026-05`.
- Filter to residential single-family properties.
- Remove duplicate listing records using listing identifiers.
- Convert key numeric fields such as `ClosePrice`, `ListPrice`, `LivingArea`, beds, baths, lot size, latitude, and longitude.
- Create `close_month` from the monthly source file for consistent time-based splitting.
- Remove or handle clearly invalid values, such as non-positive prices and very small living area values.
- Create basic features such as `property_age`, `log_living_area`, and `bathrooms_per_bedroom`.
- Save cleaned train, test, and full datasets to `outputs/`.

Main processed outputs:

- `outputs/train_preprocessed.csv`
- `outputs/test_preprocessed.csv`
- `outputs/full_preprocessed_week3.csv`

## Feature Engineering

Feature engineering is handled in `notebooks/05_feature_engineering.ipynb`.

Features tested include:

- Price-per-square-foot features.
- Lot-to-living-area ratio.
- Home age and newer-home flags.
- Bedroom and bathroom ratio features.
- Log-transformed size and days-on-market features.
- Unified school district mapping using California school district boundary data.
- County and district historical market-context features.

The feature-engineering experiments showed that larger feature sets did not always improve the linear model. Compact engineered features were competitive, but the baseline feature set remained strong.

## Models Tested

The project tested several model families:

- Linear Regression baseline.
- Decision Tree Regressor.
- Random Forest Regressor.
- XGBoost with log-transformed target.
- Streamlit-specific simple models using only app input fields.
- Streamlit-specific county-enhanced XGBoost model.

All main modeling notebooks use the same holdout month, `2026-05`, for fair comparison.

## Best Results

The strongest full modeling result came from the Week 7 XGBoost models.

Best full-feature model results on the May 2026 test set:

| Model | R2 | MAPE | MdAPE | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| XGBoost reality features | 0.6347 | 7.26% | 2.53% | $74,895 | $1,013,075 |
| XGBoost market-context features | 0.6345 | 7.18% | 2.49% | $74,596 | $1,013,250 |
| Linear Regression baseline | 0.6336 | 24.56% | 10.14% | $198,949 | $1,014,591 |

Main conclusion:

Linear Regression remains a strong and explainable baseline, but XGBoost gives much better percentage-error performance for typical homes. The XGBoost models have similar R2 values to the linear baseline, but much lower MAPE and MdAPE.

## Streamlit App

The project includes a simple Streamlit app in `app.py`.

The app has two versions:

- `Simple 4-Input Demo`: uses `LivingArea`, beds, baths, and lot size.
- `Advanced County XGBoost`: adds `CountyOrParish` and uses an XGBoost model.

The app is mainly a deployment demo. The simple four-input model is intentionally limited because it does not include location. The county-enhanced version is more realistic but still simpler than the full Week 7 XGBoost model.

App model results:

| App Model | R2 | MAPE | MdAPE |
|---|---:|---:|---:|
| County-enhanced XGBoost | 0.3660 | 30.54% | 20.22% |
| Simple four-input model | 0.3042 | 44.00% | 31.20% |

## How To Re-Run The Project

1. Clone or open the repository.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If using the project environment on the remote server, this command may be:

   ```bash
   /opt/base-uv/.venv/bin/python -m pip install -r requirements.txt
   ```

3. Place the raw CRMLS monthly files in `data/raw/`.

   Expected monthly files include:

   ```text
   CRMLSSold202506.csv
   CRMLSSold202507.csv
   CRMLSSold202508.csv
   CRMLSSold202509.csv
   CRMLSSold202510.csv
   CRMLSSold202511.csv
   CRMLSSold202512.csv
   CRMLSSold202601.csv
   CRMLSSold202602.csv
   CRMLSSold202603.csv
   CRMLSSold202604.csv
   CRMLSSold202605.csv
   ```

4. Run notebooks in this order:

   ```text
   notebooks/01_exploration.ipynb
   notebooks/02_preprocessing.ipynb
   notebooks/03_baseline_model.ipynb
   notebooks/04_model_comparison.ipynb
   notebooks/05_feature_engineering.ipynb
   notebooks/05_advanced_models.ipynb
   notebooks/06_evaluation.ipynb
   notebooks/07_streamlit_app_model.ipynb
   ```

5. Confirm the main output files were created in `outputs/`.

## How To Launch The App

After running `notebooks/07_streamlit_app_model.ipynb`, launch the Streamlit app from the repository root:

```bash
streamlit run app.py
```

If the `streamlit` command is not available, use:

```bash
python -m streamlit run app.py
```

On the remote server environment, use:

```bash
/opt/base-uv/.venv/bin/python -m streamlit run app.py
```

## Notes And Limitations

- Real estate prices are heavily influenced by location. Models with limited location information are much weaker.
- RMSE is high because California housing prices have a long right tail with luxury properties.
- MAPE and MdAPE are more useful for explaining typical percentage error.
- The Streamlit app is a demonstration tool and should not be treated as a formal appraisal system.

## Author

Mia Ma
