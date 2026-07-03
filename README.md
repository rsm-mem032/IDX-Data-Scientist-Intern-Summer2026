# IDX Data Scientist Intern Summer 2026

## Project Overview

This project predicts California residential single-family property close prices using historical CRMLS sold property data. The modeling target is `ClosePrice`.

## Data Scope

The analysis is limited to sold property observations where:

- `PropertyType = "Residential"`
- `PropertySubType = "SingleFamilyResidence"`

Raw source data should remain local and must not be committed to this repository.

## Repository Structure

```text
IDX-Data-Scientist-Intern-Summer2026/
├── README.md
├── metadata_notes.md
├── requirements.txt
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
└── data/
    └── .gitkeep
```

## Week 1 Progress

- Created the initial repository structure.
- Defined the project scope and target variable.
- Added metadata notes for key modeling columns.
- Set up a starter exploration notebook.
- Added dependency and git ignore configuration.

## Week 2 Progress

- Combined the monthly raw CRMLS sold extracts into one analysis workflow.
- Applied the residential single-family project scope filter.
- Removed duplicate listing records using listing identifiers.
- Added detailed exploratory data analysis for price levels, trends, missingness, and anomalies.
- Documented early feature relationships and follow-up modeling considerations in the notebook.

## How to Reproduce

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Register the notebook kernel if needed:

   ```bash
   python -m ipykernel install --user --name idx-data-science
   ```

4. Place raw CRMLS source data locally in `data/` or another ignored location.
5. Open `notebooks/01_exploration.ipynb` to reproduce the Week 2 exploratory analysis.

## Author

Mia Ma
