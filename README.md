# Solar Power Plant EDA & Regression Dashboard

A beginner-friendly Flask web application that performs a complete Exploratory
Data Analysis (EDA) and supervised regression on a solar power plant dataset,
showing results in a clean, responsive web interface.

## Project Description

The project loads the raw solar power plant CSV files, combines them into a
single analysis-ready dataset, computes descriptive statistics, detects
missing values / duplicates / outliers, generates charts (histograms, boxplots,
scatter plots, correlation heatmap, time patterns) and prints data-driven
findings. It also trains three regression models (Linear Regression, Ridge,
Random Forest) to predict AC power output and provides a live prediction form.
Everything runs in the browser through Flask pages.

## Dataset Description

The dataset consists of four CSV files in the `dataset/` folder:

| File | Rows | Description |
|---|---|---|
| `Plant_1_Generation_Data.csv` | 68,778 | 15-min AC/DC power & yield per inverter |
| `Plant_1_Weather_Sensor_Data.csv` | 3,182 | 15-min temperature & irradiation per sensor |
| `Plant_2_Generation_Data.csv` | 67,698 | 15-min AC/DC power & yield per inverter |
| `Plant_2_Weather_Sensor_Data.csv` | 3,259 | 15-min temperature & irradiation per sensor |

All four files are combined into one dataframe of **136,476 rows × 11 columns**:

- `DATE_TIME` (datetime) - 15-minute timestamps, 15 May - 17 June 2020
- `PLANT_ID` (categorical) - plant identifier (2 plants)
- `SOURCE_KEY` (categorical) - inverter identifier (44 inverters)
- `DC_POWER`, `AC_POWER` - DC and AC power output (kW)
- `DAILY_YIELD`, `TOTAL_YIELD` - daily / lifetime energy yield counters
- `SENSOR_KEY` (categorical) - weather sensor identifier
- `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION` - weather readings

The dataset has **no explicit target column**. For the "Target Variable
Analysis" section, `AC_POWER` (the power output of the plant) is treated as
the natural dependent variable, and this choice is clearly stated on the page.

## Technologies Used

- Python 3.11
- Flask (web framework)
- Pandas & NumPy (data handling)
- Matplotlib & Seaborn (visualisations)
- scikit-learn (regression models, preprocessing, metrics)
- joblib (model serialisation)
- HTML / CSS / JavaScript (front-end, AJAX prediction form)

## Project Structure

```
solar_eda_app/
│
├── app.py                    # Flask routes (/, /load, /eda, /regression, /predict)
├── requirements.txt
├── README.md
│
├── dataset/                  # original CSV files (never modified)
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
│
├── models/
│   ├── __init__.py
│   ├── load.py               # dataset loading & basic info helpers
│   ├── eda.py                # EDA calculations, plots, findings
│   └── regression.py         # regression training, evaluation, prediction
│
├── trained_models/           # saved sklearn pipelines (auto-created)
│   └── regression_pipeline.pkl
│
├── static/
│   ├── style.css
│   ├── plots/                # EDA generated charts (PNG)
│   └── regression_plots/     # regression generated charts (PNG)
│
└── templates/
    ├── base.html
    ├── index.html
    ├── load.html
    ├── eda.html
    └── regression.html
```

## Features

- **Home** (`/`) - project title, dataset description, buttons to load the
  dataset, run the EDA, or run the regression.
- **Load Dataset** (`/load`) - loads the combined dataset and shows its
  dimensions, data types, missing values, duplicates, unique values and a
  preview of the first/last 5 rows.
- **EDA** (`/eda`) - a complete, sectioned EDA:
  1. Dataset Information
  2. Statistical Description (mean, median, std, variance, quartiles)
  3. Missing Values (count + percentage)
  4. Duplicate Values
  5. Unique Values & categorical frequencies
  6. Numerical Analysis (histograms + KDE, boxplots)
  7. Categorical Analysis (records per inverter, power by plant)
  8. Outlier Analysis (IQR method)
  9. Correlation Matrix
  10. Correlation Heatmap
  11. Target Variable Analysis (AC_POWER)
  12. Key EDA Findings
- **Regression** (`/regression`) - supervised regression to predict AC power:
  1. Dataset Overview (target, features, sizes)
  2. Feature Selection & Leakage Prevention
  3. Data Preprocessing (ColumnTransformer: StandardScaler + OneHotEncoder)
  4. Models Trained (LinearRegression, Ridge, RandomForest)
  5. Model Comparison — Metrics table (MAE, MSE, RMSE, R², Adj. R²)
  6. Model Comparison — Bar charts
  7. Actual vs Predicted scatter plots (per model)
  8. Residuals Distribution histograms (per model)
  9. Feature Importance — RandomForest
  10. Key Findings
  11. **Live Prediction Form** — enter environmental conditions to get an
      AC_POWER prediction via AJAX (no page reload)

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
# From the project root (solar_eda_app/)
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

## EDA Techniques Used

- `head()` / `tail()` / `shape` / `dtypes` / `info()` for basic inspection
- `describe()` plus variance for numeric columns
- `isnull().sum()` for missing-value detection
- `duplicated().sum()` for duplicate detection
- `nunique()` and `value_counts()` for unique/categorical analysis
- Histograms with KDE, boxplots, scatter plots
- Pearson correlation matrix + heatmap
- IQR (1.5 × IQR) method for outlier detection
- Zero-inflation handling: solar power columns are ~50% zero (night-time), so
  IQR quartiles are computed on daytime records (values > 0)
- Time analysis: hourly average power pattern (diurnal cycle)

## Regression Techniques Used

- **Target**: `AC_POWER` (AC power output in kW)
- **Features**: IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE (numeric),
  PLANT_ID (one-hot encoded)
- **Dropped columns** (to prevent target leakage): DC_POWER (near-perfect
  correlate of AC_POWER), DAILY_YIELD, TOTAL_YIELD, SOURCE_KEY, SENSOR_KEY,
  DATE_TIME
- **Preprocessing**: `ColumnTransformer` with `StandardScaler` (numeric) and
  `OneHotEncoder(drop='first')` (categorical)
- **Train/test split**: 80/20, `random_state=42`
- **Models**: LinearRegression, Ridge (L2, α=1.0), RandomForestRegressor
  (100 trees, max_depth=15, min_samples_leaf=5)
- **Metrics**: MAE, MSE, RMSE, R², Adjusted R²
- **Visualisations**: Actual vs Predicted scatter, Residuals histogram,
  Feature importance bar chart, Model comparison grouped bars
- **Model persistence**: Best pipeline saved with `joblib` for live prediction
- **Prediction form**: AJAX POST to `/predict` with 4 inputs (ambient temp,
  module temp, irradiation, plant ID dropdown)

## Main Findings

- No missing values or duplicate rows in the raw source files. After merging
  generation with weather data, only **4 generation timestamps** lack a
  matching weather reading.
- **Zero-inflation**: `DC_POWER` / `AC_POWER` are 0 in ~49.5% of rows and
  `IRRADIATION` in ~44% - these are night-time records. All power/irradiation
  columns are therefore heavily right-skewed.
- `IRRADIATION` and `MODULE_TEMPERATURE` are very strongly correlated
  (r ≈ 0.95) - module temperature rises with sunlight.
- `DC_POWER` and `AC_POWER` are near-perfectly correlated within each plant,
  but the two plants use very different scales (Plant 1 DC max ≈ 14,471 vs
  Plant 2 ≈ 1,421), so the combined correlation is only ~0.75.
- `TOTAL_YIELD` (lifetime counter) differs hugely between plants (millions vs
  billions) - the two plants are not directly comparable.
- Inverter record counts are imbalanced (2355 to 3259 records), indicating
  missing timestamps for some inverters.
- Data covers 34 days (15 May - 17 June 2020) at 15-minute resolution.
