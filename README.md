# IDX Data Scientist Intern Summer 2026  
## California Property Close Price Prediction

## Project Overview

This project focuses on predicting the final sales price of California residential single-family properties using historical CRMLS sold property data.

The main objective is to build a reproducible machine learning workflow that can estimate a property's `ClosePrice` based on its characteristics, such as living area, number of bedrooms and bathrooms, lot size, location, and other relevant property attributes.

This project is part of the IDX Exchange Data Scientist Intern Summer 2026 program.

## Business Problem

Accurately estimating property close prices is important for real estate decision-making, pricing strategy, market analysis, and buyer/seller guidance.

In this project, the goal is to use historical sold-property records to train a predictive model that can estimate the final close price of a residential single-family property in California.

The target variable is:

- `ClosePrice`: the final sales price of the property.

## Data Source

The dataset contains historical sold property records from CRMLS, including property-level features such as:

- Living area
- Bedrooms
- Bathrooms
- Lot size
- Year built
- Location information
- Close date
- Final close price

Per project requirements, the modeling dataset is restricted to properties where:

- `PropertyType = "Residential"`
- `PropertySubType = "SingleFamilyResidence"`

Raw CSV, Excel, or source data files are not included in this repository.

## Project Workflow

The project is organized around a full machine learning workflow:

1. **Data Access and Understanding**
   - Download authorized CRMLSSold files.
   - Review metadata documentation.
   - Understand key property features and target variable.

2. **Exploratory Data Analysis**
   - Explore distributions of `ClosePrice`, `LivingArea`, bedrooms, bathrooms, and lot size.
   - Identify missing values, outliers, and data quality issues.
   - Analyze price patterns across property and location features.

3. **Data Preprocessing**
   - Filter the dataset to residential single-family properties.
   - Handle missing values.
   - Encode categorical variables.
   - Prepare training and testing datasets using a time-based split.

4. **Baseline Modeling**
   - Train a baseline linear regression model.
   - Evaluate initial performance using test-set R².

5. **Model Comparison**
   - Compare additional regression models such as decision trees and random forests.
   - Evaluate model strengths, weaknesses, and performance differences.

6. **Feature Engineering**
   - Create additional predictive features such as property age and bed/bath ratios.
   - Explore geographic features, including school district-based regional information.

7. **Advanced Modeling**
   - Test advanced models such as gradient boosting, XGBoost, or LightGBM.
   - Tune selected hyperparameters to improve predictive performance.

8. **Model Evaluation**
   - Evaluate final models using multiple metrics, including:
     - R²
     - MAPE
     - MdAPE
   - Analyze model performance across different price ranges.

9. **Documentation and Presentation**
   - Document the modeling process, assumptions, results, and reproduction steps.
   - Prepare final presentation materials for stakeholders.

## Repository Structure

```text
IDX-Data-Scientist-Intern-Summer2026/
│
├── README.md
├── metadata_notes.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── src/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
└── data/
    └── .gitkeep
