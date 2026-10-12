"""
app.py
------
Flask entry point for the Solar Power Plant EDA dashboard.

Routes
------
/             home page (project description + navigation)
/load         load & inspect the combined dataset
/eda          full exploratory data analysis
/regression   supervised regression to predict AC_POWER
/ensemble     ensemble models to predict AC_POWER
/unsupervised unsupervised learning: clustering, DR, anomaly detection
/model_analysis model performance analysis: CV, metrics, calibration, tuning
/predict      AJAX endpoint for single-row prediction (regression)
/predict_ensemble AJAX endpoint for single-row prediction (ensemble)
"""

import json
import os
import time
import uuid
from flask import Flask, jsonify, render_template, request, session

from models.load import (
    load_data,
    get_first_rows,
    get_last_rows,
    get_shape,
    get_column_names,
    get_dtypes,
    get_info,
    get_description,
    get_missing_values,
    get_duplicate_count,
    get_unique_counts,
)
from models.eda import run_full_eda
from models.regression import run_full_regression, predict_single, get_plant_ids
from models.ensemble import run_full_ensemble, predict_single as predict_single_ensemble, get_plant_ids as get_plant_ids_ensemble
from models.unsupervised import run_full_unsupervised
from models.model_analysis import run_full_model_analysis

app = Flask(__name__)
app.secret_key = 'solar_eda_secret_key_2026'

DATASET_NAME = "Solar Power Plant Generation & Weather Data"
SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.flask_sessions')
os.makedirs(SESSION_DIR, exist_ok=True)


def save_large_session(data):
    key = uuid.uuid4().hex
    path = os.path.join(SESSION_DIR, f'{key}.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    for old in os.listdir(SESSION_DIR):
        fp = os.path.join(SESSION_DIR, old)
        if time.time() - os.path.getmtime(fp) > 3600:
            os.remove(fp)
    return key


def load_large_session(key):
    path = os.path.join(SESSION_DIR, f'{key}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@app.route("/")
def index():
    """Dashboard home page."""
    return render_template("index.html", dataset_name=DATASET_NAME)


@app.route("/load")
def load_page():
    """
    Load the combined dataset and show its basic information:
    dimensions, first rows, dtypes, missing values, duplicates, uniques.
    """
    try:
        df = load_data()

        # Convert dataframes to plain Python structures for the templates.
        first_rows = df.head(5).to_dict(orient="records")
        last_rows = df.tail(5).to_dict(orient="records")
        columns = list(df.columns)

        return render_template(
            "load.html",
            dataset_name=DATASET_NAME,
            n_rows=df.shape[0],
            n_cols=df.shape[1],
            first_rows=first_rows,
            last_rows=last_rows,
            columns=columns,
            dtypes=get_dtypes(df),
            missing_counts=get_missing_values(df),
            duplicate_count=get_duplicate_count(df),
            unique_counts=get_unique_counts(df),
            memory_kb=round(df.memory_usage(deep=True).sum() / 1024, 2),
            error=None,
        )
    except FileNotFoundError as err:
        return render_template(
            "load.html",
            dataset_name=DATASET_NAME,
            n_rows=None,
            n_cols=None,
            first_rows=[],
            last_rows=[],
            columns=[],
            dtypes={},
            missing_counts={},
            duplicate_count=0,
            unique_counts={},
            memory_kb=0,
            error=str(err),
        )


@app.route("/eda")
def eda_page():
    """
    Run the complete EDA (calculations + plots) and render the results.
    Plots are written into static/plots/ on every request so the page always
    reflects the current dataset.
    """
    try:
        df = load_data()
        result = run_full_eda(df)
        return render_template("eda.html", dataset_name=DATASET_NAME, error=None, **result)
    except FileNotFoundError as err:
        return render_template("eda.html", dataset_name=DATASET_NAME, error=str(err))


@app.route("/regression")
def regression_page():
    """Train regression models and render the results page."""
    try:
        df = load_data()
        result = run_full_regression(df)
        plant_ids = get_plant_ids(df)
        return render_template(
            "regression.html",
            dataset_name=DATASET_NAME,
            plant_ids=plant_ids,
            error=None,
            **result,
        )
    except FileNotFoundError as err:
        return render_template("regression.html", dataset_name=DATASET_NAME, error=str(err))


@app.route("/ensemble")
def ensemble_page():
    """Train ensemble models and render the results page."""
    try:
        df = load_data()
        result = run_full_ensemble(df)
        plant_ids = get_plant_ids_ensemble(df)
        return render_template(
            "ensemble.html",
            dataset_name=DATASET_NAME,
            plant_ids=plant_ids,
            error=None,
            **result,
        )
    except FileNotFoundError as err:
        return render_template("ensemble.html", dataset_name=DATASET_NAME, error=str(err))


@app.route("/predict", methods=["POST"])
def predict():
    """Handle single-row prediction from the web form."""
    try:
        data = request.get_json(force=True)
        pred = predict_single(
            ambient_temp=data["ambient_temperature"],
            module_temp=data["module_temperature"],
            irradiation=data["irradiation"],
            plant_id=data["plant_id"],
        )
        return jsonify({"status": "ok", "prediction": pred})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/predict_ensemble", methods=["POST"])
def predict_ensemble():
    """Handle single-row prediction from the ensemble web form."""
    try:
        data = request.get_json(force=True)
        pred = predict_single_ensemble(
            ambient_temp=data["ambient_temperature"],
            module_temp=data["module_temperature"],
            irradiation=data["irradiation"],
            plant_id=data["plant_id"],
        )
        return jsonify({"status": "ok", "prediction": pred})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/unsupervised", methods=["GET"])
def unsupervised_page():
    """Show the unsupervised learning page.
    
    On first load: shows a 'Run Analysis' button.
    After analysis: shows the full results page.
    """
    key = session.pop('unsupervised_key', None)
    results = load_large_session(key) if key else None
    if results:
        dr = results["dr"]
        pca_loadings_table = dr["pca"].get("loadings", [])
        pca_loadings_plot = next(
            (p for p in results["plots"] if p["file"] == "pca_loadings_heatmap.png"), {}
        )
        dbscan_kdist = next(
            (p for p in results["plots"] if p["file"] == "dbscan_k_distance.png"), {}
        )
        clustering = dict(results["clustering"])
        if "dbscan" in clustering:
            clustering["dbscan"] = dict(clustering["dbscan"])
            clustering["dbscan"]["kdist_plot"] = dbscan_kdist
        return render_template(
            "unsupervised_results.html",
            clustering=clustering,
            anomaly=results["anomaly"],
            dr=dr,
            findings=results["findings"],
            n_samples=results["n_samples"],
            plots=results["plots"],
            cluster_analysis=results["cluster_analysis"],
            methodology=results["methodology"],
            pca_loadings_table=pca_loadings_table,
            pca_loadings_plot=pca_loadings_plot,
        )
    # First load: show the run button
    return render_template("unsupervised.html")


@app.route("/unsupervised/run", methods=["POST"])
def unsupervised_run():
    """Trigger unsupervised learning computation via AJAX."""
    df = load_data()
    result = run_full_unsupervised(df)
    # Store results in session for the next page load
    anomaly_session = {
        "isolation_forest": {
            k: v for k, v in result["anomaly"]["isolation_forest"].items()
            if k != "scores"
        },
        "one_class_svm": result["anomaly"]["one_class_svm"],
    }
    full_results = {
        "clustering": result["clustering"],
        "anomaly": anomaly_session,
        "dr": result["dr"],
        "findings": result["findings"],
        "n_samples": result["n_samples"],
        "plots": result["plots"],
        "cluster_analysis": result["cluster_analysis"],
        "methodology": result["methodology"],
    }
    session['unsupervised_key'] = save_large_session(full_results)
    return jsonify({"status": "ok", "redirect": "/unsupervised"})


@app.route("/model_analysis", methods=["GET"])
def model_analysis_page():
    """Show the model performance analysis page.

    On first load: shows a 'Run Analysis' button.
    After analysis: shows the full results page.
    """
    results = session.pop('model_analysis_results', None)
    if results:
        return render_template(
            "model_analysis_results.html",
            dataset_name=DATASET_NAME,
            error=None,
            **results,
        )
    return render_template("model_analysis.html", dataset_name=DATASET_NAME)


@app.route("/model_analysis/run", methods=["POST"])
def model_analysis_run():
    """Trigger model performance analysis via AJAX."""
    df = load_data()
    result = run_full_model_analysis(df)
    session['model_analysis_results'] = result
    return jsonify({"status": "ok", "redirect": "/model_analysis"})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
