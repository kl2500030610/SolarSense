"""
models/load.py
--------------
Dataset loading and basic dataset information.

The Solar Power Plant dataset is stored as four CSV files (one per plant,
split into generation and weather-sensor records). To analyse the full
dataset together, this module:

1. reads all four raw CSV files from the dataset/ folder (read-only,
   nothing is ever written back to them)
2. combines the two generation files into one dataframe
3. combines the two weather-sensor files into one dataframe
4. merges the weather data onto the generation data using
   (PLANT_ID, DATE_TIME), because each plant has exactly one weather
   sensor but many inverters

No column names or statistics are hardcoded here. Every helper derives
its result from the actual dataframe that is passed to it.
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

GENERATION_FILES = [
    "Plant_1_Generation_Data.csv",
    "Plant_2_Generation_Data.csv",
]

WEATHER_FILES = [
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_csv(path):
    """Read a CSV from disk. Raises a clear error if the file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Make sure the CSV files are inside the dataset/ folder."
        )
    return pd.read_csv(path)


def load_data():
    """
    Load and combine the full solar power plant dataset.

    Returns a single analysis-ready dataframe with these columns:
        DATE_TIME, PLANT_ID, SOURCE_KEY (inverter id), DC_POWER, AC_POWER,
        DAILY_YIELD, TOTAL_YIELD, SENSOR_KEY (weather sensor id),
        AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION
    """
    # 1. Generation data from both plants (identical schema -> concat).
    gen_frames = []
    for fname in GENERATION_FILES:
        df = _read_csv(os.path.join(DATASET_DIR, fname))
        # Plant 1 stores dates as DD-MM-YYYY (day-first); Plant 2 uses the
        # ISO YYYY-MM-DD format, so parse it without the day-first flag.
        dayfirst = fname.startswith("Plant_1")
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=dayfirst)
        gen_frames.append(df)
    generation = pd.concat(gen_frames, ignore_index=True)

    # 2. Weather data from both plants (both use ISO format).
    weather_frames = []
    for fname in WEATHER_FILES:
        df = _read_csv(os.path.join(DATASET_DIR, fname))
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
        weather_frames.append(df)
    weather = pd.concat(weather_frames, ignore_index=True)

    # The generation files and the weather files BOTH have a SOURCE_KEY
    # column, but they mean different things (inverter id vs sensor id).
    # Rename the weather one so the merge does not produce "_x"/"_y" suffixes.
    weather = weather.rename(columns={"SOURCE_KEY": "SENSOR_KEY"})

    # 3. Merge weather onto generation. 'left' keeps every generation row;
    # timestamps without a matching weather reading become NaN (reported in EDA).
    combined = generation.merge(
        weather,
        on=["PLANT_ID", "DATE_TIME"],
        how="left",
    )
    return combined


# ---------------------------------------------------------------------------
# Basic inspection helpers
# ---------------------------------------------------------------------------
def get_first_rows(df, n=5):
    """First n rows of the dataframe."""
    return df.head(n)


def get_last_rows(df, n=5):
    """Last n rows of the dataframe."""
    return df.tail(n)


def get_shape(df):
    """(rows, columns) of the dataframe."""
    return df.shape


def get_column_names(df):
    """Column names as a list."""
    return list(df.columns)


def get_dtypes(df):
    """Column names mapped to their data types."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def get_info(df):
    """Rows, columns, memory usage and non-null counts per column."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "non_null_counts": {col: int(df[col].notna().sum()) for col in df.columns},
    }


def get_description(df):
    """
    Descriptive statistics (count, mean, std, min, quartiles, max) for all
    numeric columns, returned as JSON-friendly dictionaries per column.
    """
    numeric_cols = get_numeric_cols(df)
    if not numeric_cols:
        return {}
    describe = df[numeric_cols].describe().T.round(3)
    return describe.to_dict(orient="index")


def get_missing_values(df):
    """Missing value count per column (only columns with missing values)."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    return {
        col: {
            "count": int(missing[col]),
            "percent": round(float(missing[col] / len(df) * 100), 3),
        }
        for col in missing.index
    }


def get_duplicate_count(df):
    """Number of fully duplicated rows."""
    return int(df.duplicated().sum())


def get_unique_counts(df):
    """Number of unique values per column."""
    return {col: int(df[col].nunique()) for col in df.columns}


# ---------------------------------------------------------------------------
# Column type classification (derived from the actual dataframe)
# ---------------------------------------------------------------------------
def get_numeric_cols(df):
    """Numeric (float/int) columns, excluding low-cardinality ID columns."""
    numeric_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        # Integer columns that act as identifiers (e.g. PLANT_ID with 2
        # values) are categorical in practice, not numeric features.
        if df[col].dtype.kind in "iu" and df[col].nunique() <= 20:
            continue
        numeric_cols.append(col)
    return numeric_cols


def get_categorical_cols(df):
    """Object columns plus low-cardinality integer identifier columns,
    excluding datetime columns (which are analysed separately)."""
    datetime_cols = get_datetime_cols(df)
    cat_cols = [
        c for c in df.select_dtypes(exclude=[np.number]).columns
        if c not in datetime_cols
    ]
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in cat_cols:
            continue
        if df[col].dtype.kind in "iu" and df[col].nunique() <= 20:
            cat_cols.append(col)
    return cat_cols


def get_datetime_cols(df):
    """Columns holding datetime values."""
    return [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]


if __name__ == "__main__":
    # Simple smoke test: `python models/load.py`
    data = load_data()
    print("Combined shape:", get_shape(data))
    print("Columns:", get_column_names(data))
    print("Numeric cols:", get_numeric_cols(data))
    print("Categorical cols:", get_categorical_cols(data))
    print("Datetime cols:", get_datetime_cols(data))
    print("Missing values:", get_missing_values(data))
    print("Duplicate rows:", get_duplicate_count(data))
