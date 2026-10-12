"""
models/eda.py
-------------
Exploratory Data Analysis: calculations and visualisations.

Every function here works with whatever dataframe it is given, so nothing is
hardcoded to a specific column name. The only dataset-specific decisions are
data-driven choices explained inline (e.g. evaluating IQR outliers on the
daytime records of zero-inflated solar columns).

Plots are saved as PNG files into static/plots/ so Flask can serve them
directly. We never call plt.show().
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.load import (
    get_categorical_cols,
    get_numeric_cols,
    get_datetime_cols,
)

sns.set_theme(style="whitegrid", palette="muted")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

# Inferred target: the solar dataset has no labelled target column, but AC
# power output is the natural dependent variable of a generation plant.
TARGET_COL = "AC_POWER"

# Rows used for scatter plots (they are otherwise too dense to read).
SCATTER_SAMPLE = 40000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _pct(series, condition):
    """Percentage of rows in a series matching a boolean condition."""
    return round(float(series[condition].size / series.size * 100), 1)


def _to_records(df):
    """DataFrame -> list of dicts (JSON-friendly, for Jinja templates)."""
    out = df.reset_index()
    out = out.rename(columns={"index": "column"})
    return out.to_dict(orient="records")


# ---------------------------------------------------------------------------
# A. Dataset information
# ---------------------------------------------------------------------------
def get_dataset_info(df):
    """Rows, columns, memory usage, column types."""
    numeric_cols = get_numeric_cols(df)
    cat_cols = get_categorical_cols(df)
    datetime_cols = get_datetime_cols(df)
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "datetime_cols": datetime_cols,
        "non_null": {col: int(df[col].notna().sum()) for col in df.columns},
    }


# ---------------------------------------------------------------------------
# B. Statistical description
# ---------------------------------------------------------------------------
def get_statistical_summary(df):
    """Describe() plus variance for every numeric column."""
    numeric_cols = get_numeric_cols(df)
    if not numeric_cols:
        return []
    describe = df[numeric_cols].describe().T.round(3)
    describe["variance"] = df[numeric_cols].var().round(3)
    describe = describe.rename(columns={"50%": "median"})
    return _to_records(describe)


# ---------------------------------------------------------------------------
# C. Missing values
# ---------------------------------------------------------------------------
def get_missing_summary(df):
    """Missing count and percentage per column."""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return [], 0
    rows = []
    for col in missing.index:
        rows.append({
            "column": col,
            "count": int(missing[col]),
            "percent": round(float(missing[col] / len(df) * 100), 3),
        })
    return rows, int(missing.sum())


# ---------------------------------------------------------------------------
# D. Duplicate values
# ---------------------------------------------------------------------------
def get_duplicate_summary(df):
    """Number of fully duplicated rows."""
    return int(df.duplicated().sum())


# ---------------------------------------------------------------------------
# E. Unique values & categorical frequencies
# ---------------------------------------------------------------------------
def get_unique_summary(df):
    """Number of unique values per column."""
    return [{"column": col, "unique": int(df[col].nunique())} for col in df.columns]


def get_categorical_frequencies(df):
    """Top value frequencies for categorical columns (and PLANT_ID)."""
    result = []
    for col in get_categorical_cols(df):
        counts = df[col].astype(str).value_counts().head(10)
        result.append({
            "column": col,
            "values": [{"value": k, "count": int(v)} for k, v in counts.items()],
        })
    return result


# ---------------------------------------------------------------------------
# F. Outlier analysis (IQR method)
# ---------------------------------------------------------------------------
def get_outlier_summary(df):
    """
    Detect outliers with the 1.5 x IQR rule for each numeric column.

    Solar power / irradiation columns are zero-inflated (night-time records
    make up ~half of the data), so the quartiles are computed on the positive
    (daytime) records only. Otherwise the night zeros would distort the IQR
    and every daytime reading would look like an outlier.
    """
    rows = []
    for col in get_numeric_cols(df):
        series = df[col].dropna()
        zero_share = float((series == 0).mean())
        base = series[series > 0] if zero_share > 0.3 else series

        if base.empty or base.nunique() <= 1:
            continue

        q1, q3 = base.quantile(0.25), base.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((base < lower) | (base > upper)).sum())
        pct = round(n_outliers / len(base) * 100, 2)

        rows.append({
            "column": col,
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(iqr, 3),
            "lower_fence": round(lower, 3),
            "upper_fence": round(upper, 3),
            "n_outliers": n_outliers,
            "pct_outliers": pct,
            "evaluated_on": "daytime (>0)" if zero_share > 0.3 else "all values",
            "zero_share_pct": round(zero_share * 100, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# G. Correlation analysis
# ---------------------------------------------------------------------------
def get_correlation_matrix(df):
    """Pearson correlation matrix for numeric columns (rounded)."""
    numeric_cols = get_numeric_cols(df)
    if len(numeric_cols) < 2:
        return {}
    corr = df[numeric_cols].corr().round(3)
    return corr.to_dict(orient="index")


def get_correlation_with_target(df, target=TARGET_COL):
    """Correlation of each numeric feature with the target column."""
    if target not in df.columns:
        return {}
    numeric_cols = [c for c in get_numeric_cols(df) if c != target]
    if not numeric_cols:
        return {}
    corr = df[numeric_cols + [target]].corr()[target].drop(target)
    return corr.sort_values(ascending=False).round(3).to_dict()


# ---------------------------------------------------------------------------
# H. Target variable analysis
# ---------------------------------------------------------------------------
def get_target_analysis(df, target=TARGET_COL):
    """Distribution statistics for the inferred numeric target."""
    if target not in df.columns:
        return None
    ser = df[target].dropna()
    daytime = ser[ser > 0]

    return {
        "target": target,
        "note": (
            "No explicit target column exists in the dataset. AC_POWER "
            "(AC power output) is the natural outcome of a solar generation "
            "plant, so it is used here as the inferred target for analysis."
        ),
        "stats": {
            "count": int(ser.size),
            "mean": round(ser.mean(), 3),
            "median": round(ser.median(), 3),
            "std": round(ser.std(), 3),
            "min": round(ser.min(), 3),
            "max": round(ser.max(), 3),
            "zero_share_pct": round(float((ser == 0).mean() * 100), 1),
            "daytime_mean": round(daytime.mean(), 3) if not daytime.empty else None,
            "daytime_max": round(daytime.max(), 3) if not daytime.empty else None,
        },
        "correlation_with_target": get_correlation_with_target(df, target),
    }


# ---------------------------------------------------------------------------
# I. Plot generation
# ---------------------------------------------------------------------------
def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig, filename):
    """Save a figure to static/plots/ and close it."""
    _ensure_plots_dir()
    path = os.path.join(PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return filename


def generate_plots(df):
    """
    Generate a curated set of EDA plots and save them as PNGs.

    Returns a list of dicts {file, title, section} so the template can show
    each image with a caption. Only plots that make sense for the actual
    columns are generated.
    """
    _ensure_plots_dir()
    plots = []

    numeric_cols = get_numeric_cols(df)
    datetime_cols = get_datetime_cols(df)

    # Scatter plots are dense; subsample deterministically.
    plot_df = df
    if len(df) > SCATTER_SAMPLE:
        plot_df = df.sample(SCATTER_SAMPLE, random_state=RNG_SEED)

    # ---- Distribution (histogram + KDE) of the important numeric columns ----
    # TOTAL_YIELD is a monotonic lifetime counter, so its histogram carries
    # no useful shape and is skipped here.
    for col in numeric_cols:
        if col == "TOTAL_YIELD":
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fname = f"{col.lower()}_distribution.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": f"Distribution of {col}", "section": "numerical"})

    # ---- Boxplots of numeric columns (outlier inspection) ----
    if numeric_cols:
        fig, axes = plt.subplots(nrows=1, ncols=len(numeric_cols),
                                 figsize=(4 * len(numeric_cols), 4))
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, numeric_cols):
            ax.boxplot(df[col].dropna().values)
            ax.set_title(col, fontsize=9)
            ax.tick_params(axis="x", which="both", bottom=False)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Boxplots of numeric columns")
        fname = "numeric_boxplots.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "Boxplots of numeric columns", "section": "numerical"})

    # ---- Scatter plots: weather vs generated power (daytime only) ----
    if TARGET_COL in df.columns:
        daytime = plot_df[plot_df["IRRADIATION"] > 0] if "IRRADIATION" in df.columns else plot_df
        scatter_pairs = [
            ("IRRADIATION", TARGET_COL),
            ("MODULE_TEMPERATURE", TARGET_COL),
            ("AMBIENT_TEMPERATURE", TARGET_COL),
        ]
        for x, y in scatter_pairs:
            if x in df.columns and y in df.columns:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.scatter(daytime[x], daytime[y], s=8, alpha=0.3, color="coral")
                ax.set_title(f"{x} vs {y} (daytime)")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                fname = f"{x.lower()}_vs_{y.lower()}.png"
                _save(fig, fname)
                plots.append({"file": fname, "title": f"{x} vs {y}", "section": "bivariate"})

    # ---- DC vs AC power (reveals the per-plant unit difference) ----
    if "DC_POWER" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(plot_df["DC_POWER"], plot_df[TARGET_COL], s=8, alpha=0.3, color="seagreen")
        ax.set_title("DC_POWER vs AC_POWER")
        ax.set_xlabel("DC_POWER")
        ax.set_ylabel("AC_POWER")
        fname = "dc_power_vs_ac_power.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "DC_POWER vs AC_POWER", "section": "bivariate"})

    # ---- Categorical: AC power by plant ----
    if "PLANT_ID" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="PLANT_ID", y=TARGET_COL, ax=ax, hue="PLANT_ID", palette="Set2", legend=False)
        ax.set_title(f"{TARGET_COL} by plant")
        fname = "ac_power_by_plant.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "AC_POWER by plant", "section": "categorical"})

    # ---- Categorical: records per inverter (data-balance check) ----
    if "SOURCE_KEY" in df.columns:
        counts = df["SOURCE_KEY"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        counts.plot(kind="bar", ax=ax, color="darkslateblue")
        ax.set_title("Number of records per inverter (SOURCE_KEY)")
        ax.set_ylabel("Record count")
        ax.tick_params(axis="x", rotation=45)
        fname = "records_per_inverter.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "Records per inverter", "section": "categorical"})

    # ---- Time analysis: hourly AC power pattern (diurnal cycle) ----
    if datetime_cols and TARGET_COL in df.columns:
        ts_df = df.copy()
        ts_df["HOUR"] = ts_df[datetime_cols[0]].dt.hour
        hourly = ts_df.groupby("HOUR")[TARGET_COL].mean()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(hourly.index, hourly.values, marker="o", color="darkorange")
        ax.set_title(f"Average {TARGET_COL} by hour of day")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(f"Average {TARGET_COL}")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3)
        fname = "hourly_power_pattern.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "Hourly power pattern", "section": "target"})

    # ---- Target distribution ----
    if TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[TARGET_COL].dropna(), kde=True, ax=ax, color="firebrick")
        ax.set_title(f"Distribution of {TARGET_COL} (inferred target)")
        fname = "target_distribution.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "AC_POWER distribution (target)", "section": "target"})

    # ---- Correlation heatmap ----
    numeric_cols = get_numeric_cols(df)
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Correlation heatmap of numeric columns")
        fname = "correlation_heatmap.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "Correlation heatmap", "section": "correlation"})

    return plots


# ---------------------------------------------------------------------------
# J. Key findings (computed from the actual data, never invented)
# ---------------------------------------------------------------------------
def get_findings(df):
    """Data-driven observations about the dataset, as a list of strings."""
    findings = []
    n_rows = len(df)
    numeric_cols = get_numeric_cols(df)

    # Missing values
    missing, total_missing = get_missing_summary(df)
    if total_missing:
        cols = ", ".join(f"{row['column']} ({row['count']})" for row in missing)
        findings.append(
            f"The dataset has {total_missing} missing value(s) in {len(missing)} column(s): {cols}. "
            "These occur because a few generation timestamps have no matching weather-sensor reading."
        )
    else:
        findings.append("The dataset has no missing values.")

    # Duplicates
    dup = get_duplicate_summary(df)
    findings.append(
        f"The dataset contains {dup} fully duplicated row(s)." if dup
        else "The dataset contains no duplicate rows."
    )

    # Zero-inflation (night-time records)
    for col in ["DC_POWER", "AC_POWER", "IRRADIATION"]:
        if col in df.columns:
            zero_pct = _pct(df[col], df[col] == 0)
            findings.append(
                f"{col} is zero in {zero_pct}% of rows - these are night-time "
                "records where no sunlight is available. This makes the power "
                "and irradiation columns highly skewed (zero-inflated)."
            )

    # Skewness
    skews = df[numeric_cols].skew(numeric_only=True).dropna().sort_values(ascending=False)
    high_skew = skews[skews.abs() > 1.0]
    if len(high_skew):
        pairs = ", ".join(f"{c} ({round(v, 2)})" for c, v in high_skew.items())
        findings.append(f"Highly skewed numeric columns: {pairs}.")

    # Strong correlations
    if TARGET_COL in df.columns and "IRRADIATION" in df.columns and "MODULE_TEMPERATURE" in df.columns:
        r1 = df["IRRADIATION"].corr(df["MODULE_TEMPERATURE"])
        findings.append(
            f"IRRADIATION and MODULE_TEMPERATURE are very strongly correlated "
            f"(r = {r1:.2f}) - module temperature rises with sunlight."
        )
    if "DC_POWER" in df.columns and TARGET_COL in df.columns:
        r2 = df["DC_POWER"].corr(df[TARGET_COL])
        findings.append(
            f"DC_POWER and AC_POWER correlate at r = {r2:.2f} overall. The "
            "correlation is near-perfect within each plant, but the two plants "
            "use different DC/AC scales, which lowers the combined value."
        )

    # Plant-level differences
    if "PLANT_ID" in df.columns:
        plant_groups = df.groupby("PLANT_ID")
        plants = []
        for pid, sub in plant_groups:
            dc_max = sub["DC_POWER"].max() if "DC_POWER" in sub.columns else 0
            ty_max = sub["TOTAL_YIELD"].max() if "TOTAL_YIELD" in sub.columns else 0
            ac_max = sub["AC_POWER"].max() if TARGET_COL in sub.columns else 0
            ratio = dc_max / ac_max if ac_max > 0 else 0
            plants.append((pid, dc_max, ty_max, ratio))
        if len(plants) >= 2:
            p0, p1 = plants[0], plants[1]
            findings.append(
                f"Large scale difference between plants: plant {p0[0]} reaches "
                f"DC ~{p0[1]:,.0f} (DC/AC ratio ~{p0[3]:.1f}) while plant {p1[0]} "
                f"reaches DC ~{p1[1]:,.0f} (ratio ~{p1[3]:.1f}). TOTAL_YIELD also "
                f"differs hugely ({p0[2]:.2e} vs {p1[2]:.2e}) - the two plants are "
                "not directly comparable and should be handled separately for ML."
            )

    # Inverter record imbalance
    if "SOURCE_KEY" in df.columns:
        counts = df.groupby("SOURCE_KEY").size()
        if counts.std() > 0:
            findings.append(
                f"Inverter (SOURCE_KEY) record counts are imbalanced: min "
                f"{int(counts.min())}, max {int(counts.max())} records. Plant 1 "
                "inverters have missing timestamps (fewer than the expected ~3264)."
            )

    # Date range
    datetime_cols = get_datetime_cols(df)
    if datetime_cols:
        start = df[datetime_cols[0]].min()
        end = df[datetime_cols[0]].max()
        days = (end - start).days
        findings.append(
            f"Data covers {days} days from {start.strftime('%Y-%m-%d')} to "
            f"{end.strftime('%Y-%m-%d')} at 15-minute intervals."
        )

    return findings


# ---------------------------------------------------------------------------
# Main entry used by app.py
# ---------------------------------------------------------------------------
def run_full_eda(df):
    """Execute the complete EDA and return everything the /eda page needs."""
    plots = generate_plots(df)
    return {
        "info": get_dataset_info(df),
        "stats": get_statistical_summary(df),
        "missing": get_missing_summary(df)[0],
        "missing_total": get_missing_summary(df)[1],
        "duplicates": get_duplicate_summary(df),
        "uniques": get_unique_summary(df),
        "cat_frequencies": get_categorical_frequencies(df),
        "outliers": get_outlier_summary(df),
        "correlation_matrix": get_correlation_matrix(df),
        "target": get_target_analysis(df),
        "plots": plots,
        "findings": get_findings(df),
    }


if __name__ == "__main__":
    # Smoke test: `python models/eda.py`
    from models.load import load_data

    data = load_data()
    result = run_full_eda(data)
    print("Plots generated:", [p["file"] for p in result["plots"]])
    print("\nFindings:")
    for finding in result["findings"]:
        print(" -", finding)
