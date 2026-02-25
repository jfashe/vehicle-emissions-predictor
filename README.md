# Vehicle CO2 Emissions Predictor

A supervised machine learning pipeline that predicts vehicle CO2 emissions (in g/km) from manufacturer specifications. Built with scikit-learn using a  preprocessing and modeling pipeline.
Followed a tutorial by [codewithjosh](https://www.youtube.com/watch?v=777Qb0gHuJU).

---

## 📊 MODEL PERFORMANCE

| Metric | Score |
|--------|-------|
| R² Score | **0.973** |
| Root Mean Squared Error | **10.41 g/km** |
| Mean Absolute Error | **3.13 g/km** |

> The model explains **97.3% of the variance** in CO2 emissions on unseen test data, with an average prediction error of just 3.13 g/km.

---

## Overview

Given various vehicle features (such as make, model, engine size, etc), this model predicts its CO2 output in grams per kilometer. The pipeline handles all preprocessing automatically, like missing value imputation, feature scaling, and categorical encoding, before passing data to a random forest regressor.

---

## Pipeline Architecture

```
Raw CSV Data
     │
     ▼
ColumnTransformer
├── Numerical Pipeline
│   ├── SimpleImputer     (fill missing values with column mean)
│   └── StandardScaler    (normalise to zero mean, unit variance)
└── Categorical Pipeline
    ├── SimpleImputer     (fill missing values with most frequent)
    └── OneHotEncoder     (convert text labels to binary columns)
     │
     ▼
RandomForestRegressor
     │
     ▼
CO2 Prediction (g/km)
```

---

## Features used

**Numerical:**
- Model Year
- Engine Size (L)
- Cylinders
- Fuel Consumption — City (L/100 km)
- Fuel Consumption — Highway (L/100 km)
- Fuel Consumption — Combined (L/100 km)
- Smog Level

**Categorical:**
- Make
- Model
- Vehicle Class
- Transmission

**Target:**
- CO2 Emissions (g/km)

---

The trained model will be saved as `vehicle_emission_pipeline.joblib`.

---

## TOOLS USED

- **Python 3**
- **pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **scikit-learn** — pipeline, preprocessing, model training and evaluation. heaviest hitter
- **joblib** — model serialisation

---

## 📌 Notes

- Train/test split: 80/20 with `random_state=42` for reproducibility
- The saved `.joblib` file contains the full pipeline (preprocessor + model), so it can be loaded and used for inference without retraining
- `handle_unknown="ignore"` in the encoder ensures the model won't crash on unseen vehicle makes or models at inference time



Started February 25, 9:22am -
Completed February 25, 11:04am