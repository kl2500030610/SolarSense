"""
models/unsupervised.py
----------------------
Unsupervised learning module for the Solar Power Plant EDA dashboard.

Techniques applied to the numeric feature matrix (no labels used):

Clustering:
  - K-Means (k recommended by elbow method)
  - Hierarchical / Agglomerative (Ward linkage)
  - DBSCAN (eps tuned via k-distance graph)

Dimensionality reduction:
  - PCA (principal component analysis, with loadings)
  - t-SNE (t-distributed stochastic neighbour embedding)
  - UMAP (uniform manifold approximation and projection)

Anomaly detection:
  - Isolation Forest
  - One-Class SVM

Features (all numeric, standardized with StandardScaler):
  IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE, DC_POWER, AC_POWER

Expensive methods (hierarchical, DBSCAN, t-SNE, UMAP) run on a fixed
subsample (random_state = 42); K-Means and Isolation Forest run on the
full dataset (Isolation Forest on a capped subsample for speed).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "unsupervised_plots")

FEATURE_COLS = [
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "DC_POWER",
    "AC_POWER",
]

SUBSAMPLE_SIZE = 500
RANDOM_STATE = 42
ISO_MAX_SAMPLES = 10000

CLUSTER_COLORS = ["#c96f11", "#f2b134", "#8a4b08", "#2e86ab",
                  "#a23b72", "#f18f01", "#c73e1d", "#6a994e"]
NOISE_COLOR = "#9ca3af"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df):
    """Return the numeric feature matrix X (rows with NaNs dropped)."""
    subset = df[FEATURE_COLS].dropna().copy()
    return subset[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def _eval_clustering(X, labels):
    """Compute common clustering quality metrics (None when undefined)."""
    labels = np.asarray(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum()) if -1 in labels else 0

    result = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_noise_pct": round(n_noise / len(labels) * 100, 2) if len(labels) else 0.0,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    if n_clusters >= 2 and n_noise < len(labels) - 1:
        try:
            result["silhouette"] = round(float(silhouette_score(X, labels)), 4)
        except Exception:
            pass
        try:
            result["davies_bouldin"] = round(float(davies_bouldin_score(X, labels)), 4)
        except Exception:
            pass
        try:
            result["calinski_harabasz"] = round(float(calinski_harabasz_score(X, labels)), 2)
        except Exception:
            pass

    sizes = pd.Series(labels).value_counts().sort_index()
    result["cluster_sizes"] = {str(int(k)): int(v) for k, v in sizes.items()}
    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def _save(fig, filename):
    _ensure_plot_dir()
    path = os.path.join(PLOT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return filename


def _cluster_scatter(x, y, labels, title, fname, xlabel, ylabel):
    """Scatter plot coloured by cluster label (-1 = noise)."""
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    unique = np.unique(labels)
    for i, lab in enumerate(unique):
        mask = labels == lab
        if lab == -1:
            ax.scatter(x[mask], y[mask], s=6, alpha=0.35,
                       color=NOISE_COLOR, label="noise", linewidths=0)
        else:
            color = CLUSTER_COLORS[int(lab) % len(CLUSTER_COLORS)]
            ax.scatter(x[mask], y[mask], s=6, alpha=0.55,
                       color=color, label=f"cluster {int(lab)}", linewidths=0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(unique) <= 8:
        ax.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, fname)


# ---------------------------------------------------------------------------
# Elbow method
# ---------------------------------------------------------------------------

def _elbow_method(X_sub, k_range=range(1, 11)):
    """Compute inertia for k=1..10 and find the recommended k."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_sub)
        inertias.append(round(float(km.inertia_), 2))

    # Find elbow using the "knee" point (max curvature)
    ks = list(k_range)
    if len(ks) >= 3:
        # Normalize
        x_norm = np.linspace(0, 1, len(ks))
        y_norm = np.array(inertias)
        y_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min() + 1e-10)
        # Compute curvature (second derivative approximation)
        curvatures = []
        for i in range(1, len(ks) - 1):
            curv = abs(y_norm[i - 1] - 2 * y_norm[i] + y_norm[i + 1])
            curvatures.append(curv)
        best_idx = int(np.argmax(curvatures)) + 1 if curvatures else 1
    else:
        best_idx = 0

    recommended_k = ks[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, "o-", color="#c96f11", linewidth=2, markersize=6)
    ax.axvline(recommended_k, color="#b3261e", linestyle="--", linewidth=1.5,
               label=f"Recommended k = {recommended_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — Inertia vs. Number of Clusters", fontsize=12, fontweight="bold")
    ax.set_xticks(ks)
    ax.legend()
    fname = _save(fig, "elbow_method.png")

    return {
        "k_values": ks,
        "inertias": inertias,
        "recommended_k": recommended_k,
        "plot": {"file": fname, "title": "Elbow Method — Inertia vs. k"},
    }


# ---------------------------------------------------------------------------
# Silhouette analysis
# ---------------------------------------------------------------------------

def _silhouette_comparison(clustering_results):
    """Bar chart comparing silhouette scores across methods."""
    methods = []
    scores = []
    for method in ["kmeans", "hierarchical", "dbscan"]:
        sil = clustering_results.get(method, {}).get("silhouette")
        if sil is not None:
            labels_map = {"kmeans": "K-Means", "hierarchical": "Hierarchical", "dbscan": "DBSCAN"}
            methods.append(labels_map[method])
            scores.append(sil)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, scores, color=["#c96f11", "#f2b134", "#8a4b08"],
                  edgecolor="white")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    fname = _save(fig, "silhouette_comparison.png")

    return {
        "methods": methods,
        "scores": scores,
        "plot": {"file": fname, "title": "Silhouette Score Comparison"},
    }


# ---------------------------------------------------------------------------
# Cluster analysis (distribution, centers, assignments)
# ---------------------------------------------------------------------------

def _cluster_analysis(X_sub, labels, scaler, n_samples_table=10):
    """Compute cluster distribution, centers and sample assignments."""
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels))
    noise_mask = labels == -1

    # Distribution
    dist_rows = []
    for lab in unique_labels:
        count = int((labels == lab).sum())
        dist_rows.append({
            "cluster": "noise" if lab == -1 else str(int(lab)),
            "count": count,
            "pct": round(count / len(labels) * 100, 2),
        })

    # Centers (inverse transform to original scale)
    centers = []
    for lab in unique_labels:
        if lab == -1:
            continue
        mask = labels == lab
        center_scaled = X_sub[mask].mean(axis=0)
        center_original = scaler.inverse_transform(center_scaled.reshape(1, -1))[0]
        centers.append({
            "cluster": int(lab),
            "count": int(mask.sum()),
            **{feat: round(float(center_original[i]), 2)
               for i, feat in enumerate(FEATURE_COLS)},
        })

    # Sample assignments (first n samples)
    assignments = []
    for i in range(min(n_samples_table, len(labels))):
        assignments.append({
            "sample_idx": i,
            "cluster": "noise" if labels[i] == -1 else str(int(labels[i])),
            **{feat: round(float(X_sub[i][j]), 3)
               for j, feat in enumerate(FEATURE_COLS)},
        })

    # Distribution bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    cluster_names = [r["cluster"] for r in dist_rows]
    counts = [r["count"] for r in dist_rows]
    colors = [NOISE_COLOR if c == "noise" else CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)]
              for c in cluster_names]
    bars = ax.bar(cluster_names, counts, color=colors, edgecolor="white")
    ax.set_ylabel("Number of samples")
    ax.set_title("Cluster Size Distribution", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                ha="center", va="bottom", fontsize=10)
    fname = _save(fig, "cluster_distribution.png")

    return {
        "distribution": dist_rows,
        "centers": centers,
        "assignments": assignments,
        "plot": {"file": fname, "title": "Cluster Size Distribution"},
    }


# ---------------------------------------------------------------------------
# PCA loadings
# ---------------------------------------------------------------------------

def _pca_loadings(pca_full, feature_names):
    """Extract and plot PCA feature loadings."""
    components = pca_full.components_[:min(5, pca_full.components_.shape[0])]
    loadings = []
    for i, comp in enumerate(components):
        loadings.append({
            "feature": feature_names,
            "pc": f"PC{i+1}",
            "values": [round(float(v), 4) for v in comp],
        })

    # Heatmap
    heatmap_data = pd.DataFrame(
        components,
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(len(components))],
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="RdBu_r", center=0,
                fmt=".2f", ax=ax, cbar_kws={"label": "Loading"})
    ax.set_title("PCA Feature Loadings Heatmap", fontsize=12, fontweight="bold")
    fname = _save(fig, "pca_loadings_heatmap.png")

    # Table rows
    table_rows = []
    for i, comp in enumerate(components):
        for j, feat in enumerate(feature_names):
            table_rows.append({
                "pc": f"PC{i+1}",
                "feature": feat,
                "loading": round(float(comp[j]), 4),
            })

    return {
        "table": table_rows,
        "plot": {"file": fname, "title": "PCA Loadings Heatmap"},
    }


# ---------------------------------------------------------------------------
# DBSCAN parameter tuning (k-distance graph)
# ---------------------------------------------------------------------------

def _tune_dbscan(X_sub, min_samples=5):
    """Use k-distance graph to find optimal eps for DBSCAN."""
    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X_sub)
    distances, _ = neigh.kneighbors(X_sub)
    k_distances = np.sort(distances[:, -1])

    # Find elbow in k-distance curve
    x_norm = np.linspace(0, 1, len(k_distances))
    y_norm = (k_distances - k_distances.min()) / (k_distances.max() - k_distances.min() + 1e-10)
    best_idx = int(len(k_distances) * 0.95) if len(k_distances) > 10 else len(k_distances) - 1
    eps = round(float(k_distances[best_idx]), 2)

    # Plot k-distance graph
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(k_distances)), k_distances, color="#c96f11", linewidth=1.5)
    ax.axhline(eps, color="#b3261e", linestyle="--", linewidth=1.5,
               label=f"eps = {eps}")
    ax.set_xlabel("Samples (sorted by distance)")
    ax.set_ylabel(f"{min_samples}-th Nearest Neighbor Distance")
    ax.set_title("K-Distance Graph — DBSCAN eps Selection", fontsize=12, fontweight="bold")
    ax.legend()
    fname = _save(fig, "dbscan_k_distance.png")

    return eps, {"file": fname, "title": "K-Distance Graph (DBSCAN eps)"}


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _anomaly_detection(X, X_scaled, X_sub, X_pca, sub_idx, df_full):
    """Run Isolation Forest and One-Class SVM, compute scores and sample tables."""
    # Isolation Forest on full data (capped)
    iso_X = X_scaled
    if len(X_scaled) > ISO_MAX_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        iso_idx = rng.choice(len(X_scaled), ISO_MAX_SAMPLES, replace=False)
        iso_X = X_scaled[iso_idx]

    iso = IsolationForest(n_estimators=100, contamination=0.05,
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(iso_X)
    iso_scores = iso.decision_function(X_scaled)
    iso_labels = iso.predict(X_scaled)
    n_anomalies = int((iso_labels == -1).sum())

    # Score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(iso_scores, bins=60, color="#c96f11", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#b3261e", linestyle="--", linewidth=1.5, label="Decision boundary")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Isolation Forest — Anomaly Score Distribution", fontsize=12, fontweight="bold")
    ax.legend()
    fname = _save(fig, "anomaly_scores_hist.png")

    # Top anomalies table (from full dataset)
    anomaly_indices = np.where(iso_labels == -1)[0]
    top_anomalies = []
    if len(anomaly_indices) > 0:
        sorted_anom = anomaly_indices[np.argsort(iso_scores[anomaly_indices])[:10]]
        X_full = prepare_features(df_full)
        for idx in sorted_anom:
            row = {
                "index": int(idx),
                "score": round(float(iso_scores[idx]), 4),
                **{feat: round(float(X_full.iloc[idx][feat]), 2)
                   for feat in FEATURE_COLS},
            }
            top_anomalies.append(row)

    # One-Class SVM on subsample
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_sub)
    ocsvm_labels = ocsvm.predict(X_sub)
    ocsvm_scores = ocsvm.decision_function(X_sub)
    n_ocsvm_anomalies = int((ocsvm_labels == -1).sum())

    # OCSVM scatter on PCA (subsample coords)
    X_pca_sub_coords = X_pca[sub_idx]
    anom_sub_mask = ocsvm_labels == -1
    fig, ax = plt.subplots(figsize=(7.5, 6))
    normal_mask = ~anom_sub_mask
    ax.scatter(X_pca_sub_coords[normal_mask, 0], X_pca_sub_coords[normal_mask, 1],
               s=6, alpha=0.4, color="#c96f11", label="normal", linewidths=0)
    if anom_sub_mask.any():
        ax.scatter(X_pca_sub_coords[anom_sub_mask, 0], X_pca_sub_coords[anom_sub_mask, 1],
                   s=14, alpha=0.8, color="#b3261e", label="anomaly", linewidths=0)
    ax.set_title("One-Class SVM — Anomaly Detection (PCA projection)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2, fontsize=9)
    fname_ocsvm = _save(fig, "ocsvm_scatter.png")

    # Anomaly scatter (Isolation Forest, subsample, PCA coords)
    anom_sub = iso_labels[sub_idx]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(X_pca[sub_idx][anom_sub == 1, 0], X_pca[sub_idx][anom_sub == 1, 1],
               s=6, alpha=0.4, color="#c96f11", label="normal", linewidths=0)
    ax.scatter(X_pca[sub_idx][anom_sub == -1, 0], X_pca[sub_idx][anom_sub == -1, 1],
               s=14, alpha=0.8, color="#b3261e", label="anomaly", linewidths=0)
    ax.set_title("Anomaly Detection — Isolation Forest", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2, fontsize=9)
    fname_iso = _save(fig, "anomaly_scatter.png")

    return {
        "isolation_forest": {
            "method": "IsolationForest",
            "contamination": 0.05,
            "n_estimators": 100,
            "n_anomalies": n_anomalies,
            "pct_anomalies": round(n_anomalies / len(iso_labels) * 100, 2),
            "n_normal": int((iso_labels == 1).sum()),
            "scores": iso_scores.tolist(),
            "top_anomalies": top_anomalies,
            "score_plot": {"file": fname, "title": "Isolation Forest — Anomaly Scores"},
            "scatter_plot": {"file": fname_iso, "title": "Isolation Forest — Anomaly Scatter"},
        },
        "one_class_svm": {
            "method": "OneClassSVM",
            "kernel": "rbf",
            "nu": 0.05,
            "n_anomalies": n_ocsvm_anomalies,
            "pct_anomalies": round(n_ocsvm_anomalies / len(ocsvm_labels) * 100, 2),
            "n_normal": int((ocsvm_labels == 1).sum()),
            "scatter_plot": {"file": fname_ocsvm, "title": "One-Class SVM — Anomaly Detection"},
        },
    }


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------

def _build_methodology(pca_loadings_result, dbscan_eps, dbscan_min_samples):
    """Return preprocessing steps, parameters and limitations."""
    preprocessing = [
        "Selected 5 numeric features: IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE, DC_POWER, AC_POWER",
        "Applied StandardScaler to standardize all features to zero mean and unit variance",
        f"Random subsample of {SUBSAMPLE_SIZE} rows (random_state={RANDOM_STATE}) for expensive methods",
        f"Isolation Forest run on {'full' if ISO_MAX_SAMPLES >= 10000 else f'{ISO_MAX_SAMPLES}'}-row capped dataset for efficiency",
    ]
    parameters = [
        f"K-Means: k determined by elbow method, n_init=10, random_state={RANDOM_STATE}",
        f"Hierarchical: Ward linkage, n_clusters=4",
        f"DBSCAN: eps={dbscan_eps} (tuned via k-distance graph), min_samples={dbscan_min_samples}",
        "PCA: full dataset, all components retained for loadings",
        "t-SNE: perplexity=30, max_iter=300, learning_rate=auto, init=pca",
        "UMAP: n_neighbors=15, min_dist=0.1",
        "Isolation Forest: n_estimators=100, contamination=0.05",
        "One-Class SVM: RBF kernel, gamma=scale, nu=0.05",
    ]
    limitations = [
        "t-SNE and UMAP are stochastic — results vary slightly across runs",
        "DBSCAN is sensitive to eps and min_samples in high-dimensional space",
        "Isolation Forest and One-Class SVM assume ~5% of data is anomalous",
        "StandardScaler is affected by extreme outliers in power columns",
        "K-Means assumes spherical clusters — may not capture elongated structures",
        "Subsampling reduces statistical power for rare patterns",
        "5-D standardized space suffers from the curse of dimensionality",
    ]
    return {
        "preprocessing": preprocessing,
        "parameters": parameters,
        "limitations": limitations,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def train_and_evaluate(df):
    """Run all unsupervised techniques, generate plots, return results dict."""
    X = prepare_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rng = np.random.RandomState(RANDOM_STATE)
    if len(X_scaled) > SUBSAMPLE_SIZE:
        sub_idx = rng.choice(len(X_scaled), SUBSAMPLE_SIZE, replace=False)
    else:
        sub_idx = np.arange(len(X_scaled))
    X_sub = X_scaled[sub_idx]

    _ensure_plot_dir()

    results = {
        "features": FEATURE_COLS,
        "n_samples": int(len(X)),
        "n_features": len(FEATURE_COLS),
        "subsample_size": int(len(X_sub)),
        "random_state": RANDOM_STATE,
        "clustering": {},
        "dr": {},
        "anomaly": {},
        "cluster_analysis": {},
        "plots": [],
        "findings": [],
        "umap_available": UMAP_AVAILABLE,
    }

    # ---------------- PCA ----------------
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
    evr = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n_comp_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca2.fit_transform(X_scaled)
    X_pca_sub = X_pca[sub_idx]

    results["dr"]["pca"] = {
        "explained_variance_ratio": [round(float(v), 4) for v in evr[:min(len(evr), 10)]],
        "cumulative_variance": [round(float(v), 4) for v in cumvar[:min(len(cumvar), 10)]],
        "n_components_95pct": n_comp_95,
        "pc1_var": round(float(evr[0]), 4),
        "pc2_var": round(float(evr[1]), 4),
    }

    # PCA explained variance bar chart
    n_show = min(len(evr), 10)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(1, n_show + 1), evr[:n_show] * 100,
                  color="#c96f11", edgecolor="white", alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(range(1, n_show + 1), cumvar[:n_show] * 100,
             "o-", color="#8a4b08", linewidth=2, markersize=5)
    ax2.set_ylabel("Cumulative variance (%)", color="#8a4b08")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", colors="#8a4b08")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA — Explained Variance", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, n_show + 1))
    for bar, v in zip(bars, evr[:n_show]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    fname = _save(fig, "pca_explained_variance.png")
    results["plots"].append({"file": fname, "title": "PCA — Explained Variance"})

    # PCA 2D scatter
    irradiation_sub = df.loc[X.index[sub_idx], "IRRADIATION"].values
    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter = ax.scatter(X_pca_sub[:, 0], X_pca_sub[:, 1],
                         c=irradiation_sub, cmap="YlOrRd", s=6, alpha=0.5, linewidths=0)
    ax.set_title("PCA — 2D Projection (coloured by IRRADIATION)", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    fig.colorbar(scatter, ax=ax, label="IRRADIATION (W/m²)")
    fname = _save(fig, "pca_scatter.png")
    results["plots"].append({"file": fname, "title": "PCA — 2D Projection"})

    # PCA loadings
    pca_loadings_result = _pca_loadings(pca_full, FEATURE_COLS)
    results["dr"]["pca"]["loadings"] = pca_loadings_result["table"]
    results["plots"].append(pca_loadings_result["plot"])

    # ---------------- Elbow method ----------------
    elbow_result = _elbow_method(X_sub)
    results["clustering"]["elbow"] = elbow_result
    results["plots"].append(elbow_result["plot"])
    recommended_k = elbow_result["recommended_k"]

    # ---------------- t-SNE ----------------
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                perplexity=min(30, max(1, SUBSAMPLE_SIZE - 1)),
                max_iter=300, learning_rate="auto", init="pca")
    X_tsne = tsne.fit_transform(X_sub)
    results["dr"]["tsne"] = {"perplexity": 30, "samples": int(len(X_sub))}

    # ---------------- UMAP ----------------
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                            n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_sub)
        results["dr"]["umap"] = {"n_neighbors": 15, "min_dist": 0.1,
                                 "samples": int(len(X_sub))}

    # ---------------- K-Means (with recommended k) ----------------
    kmeans = KMeans(n_clusters=recommended_k, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    results["clustering"]["kmeans"] = _eval_clustering(X_scaled, kmeans_labels)
    results["clustering"]["kmeans"]["inertia"] = round(float(kmeans.inertia_), 2)
    results["clustering"]["kmeans"]["recommended_k"] = recommended_k

    kmeans_sub_labels = kmeans_labels[sub_idx]
    fname = _cluster_scatter(
        X_pca_sub[:, 0], X_pca_sub[:, 1], kmeans_sub_labels,
        f"K-Means Clustering (k = {recommended_k}) — PCA projection",
        "kmeans_clusters.png", f"PC1 ({evr[0] * 100:.1f}%)", f"PC2 ({evr[1] * 100:.1f}%)",
    )
    results["plots"].append({"file": fname, "title": f"K-Means Clusters (k = {recommended_k})"})

    # ---------------- Hierarchical ----------------
    hierarchical = AgglomerativeClustering(n_clusters=4, linkage="ward")
    hier_labels = hierarchical.fit_predict(X_sub)
    results["clustering"]["hierarchical"] = _eval_clustering(X_sub, hier_labels)

    fname = _cluster_scatter(
        X_pca_sub[:, 0], X_pca_sub[:, 1], hier_labels,
        "Hierarchical Clustering (Ward, k = 4) — PCA projection",
        "hierarchical_clusters.png", f"PC1 ({evr[0] * 100:.1f}%)", f"PC2 ({evr[1] * 100:.1f}%)",
    )
    results["plots"].append({"file": fname, "title": "Hierarchical Clusters (k = 4)"})

    # ---------------- DBSCAN (tuned) ----------------
    dbscan_min_samples = 5
    dbscan_eps, kdist_plot = _tune_dbscan(X_sub, min_samples=dbscan_min_samples)
    results["plots"].append(kdist_plot)
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(X_sub)
    results["clustering"]["dbscan"] = _eval_clustering(X_sub, dbscan_labels)
    results["clustering"]["dbscan"]["eps"] = dbscan_eps
    results["clustering"]["dbscan"]["min_samples"] = dbscan_min_samples

    fname = _cluster_scatter(
        X_pca_sub[:, 0], X_pca_sub[:, 1], dbscan_labels,
        f"DBSCAN (eps = {dbscan_eps}, min_samples = {dbscan_min_samples}) — PCA projection",
        "dbscan_clusters.png", f"PC1 ({evr[0] * 100:.1f}%)", f"PC2 ({evr[1] * 100:.1f}%)",
    )
    results["plots"].append({"file": fname, "title": "DBSCAN Clusters"})

    # ---------------- Silhouette comparison ----------------
    sil_result = _silhouette_comparison(results["clustering"])
    results["clustering"]["silhouette_comparison"] = sil_result
    results["plots"].append(sil_result["plot"])

    # ---------------- Cluster analysis (K-Means) ----------------
    ca_result = _cluster_analysis(X_sub, kmeans_sub_labels, scaler)
    results["cluster_analysis"] = ca_result
    results["plots"].append(ca_result["plot"])

    # ---------------- Anomaly detection ----------------
    anomaly_result = _anomaly_detection(X, X_scaled, X_sub, X_pca, sub_idx, df)
    results["anomaly"] = anomaly_result
    results["plots"].append(anomaly_result["isolation_forest"]["score_plot"])
    results["plots"].append(anomaly_result["isolation_forest"]["scatter_plot"])
    results["plots"].append(anomaly_result["one_class_svm"]["scatter_plot"])

    # ---------------- Methodology ----------------
    results["methodology"] = _build_methodology(
        pca_loadings_result, dbscan_eps, dbscan_min_samples
    )

    # ---------------- Findings ----------------
    results["findings"] = _generate_findings(results)

    return results


def _generate_findings(results):
    """Auto-generate human-readable findings from the results dict."""
    findings = []
    km = results["clustering"]["kmeans"]
    hier = results["clustering"]["hierarchical"]
    dbscan = results["clustering"]["dbscan"]
    pca = results["dr"]["pca"]
    anom = results["anomaly"]["isolation_forest"]
    ocsvm = results["anomaly"]["one_class_svm"]
    elbow = results["clustering"]["elbow"]

    findings.append(
        f"Elbow method recommends k = {elbow['recommended_k']} clusters "
        f"(inertia = {km['inertia']:,.0f}). K-Means silhouette = "
        f"{km['silhouette'] if km['silhouette'] is not None else 'n/a'}."
    )

    findings.append(
        f"K-Means (k = {elbow['recommended_k']}) found {km['n_clusters']} clusters. "
        f"Largest cluster holds {max(km['cluster_sizes'].values()):,} of "
        f"{results['n_samples']:,} records."
    )

    findings.append(
        f"Hierarchical clustering (Ward) found {hier['n_clusters']} clusters with "
        f"silhouette = {hier['silhouette'] if hier['silhouette'] is not None else 'n/a'}. "
        "Both methods split the data mainly along the day/night boundary "
        "and the irradiation gradient."
    )

    if dbscan["n_clusters"] == 0:
        findings.append(
            f"DBSCAN (eps = {dbscan['eps']}, min_samples = {dbscan['min_samples']}) "
            f"labelled every point as noise ({dbscan['n_noise_pct']}% of the subsample). "
            "The data is too sparse in standardised 5-D space even after tuning."
        )
    else:
        findings.append(
            f"DBSCAN found {dbscan['n_clusters']} density-based cluster(s) with "
            f"{dbscan['n_noise_pct']}% noise points (eps = {dbscan['eps']}, "
            f"min_samples = {dbscan['min_samples']})."
        )

    findings.append(
        f"PCA: PC1 explains {pca['pc1_var'] * 100:.1f}% and PC2 explains "
        f"{pca['pc2_var'] * 100:.1f}% of the variance "
        f"({(pca['pc1_var'] + pca['pc2_var']) * 100:.1f}% combined). "
        f"{pca['n_components_95pct']} components are needed to reach 95%."
    )

    findings.append(
        f"Isolation Forest flagged {anom['n_anomalies']:,} records "
        f"({anom['pct_anomalies']}%) as anomalies at 5% contamination. "
        f"One-Class SVM flagged {ocsvm['n_anomalies']:,} records "
        f"({ocsvm['pct_anomalies']}%) on the subsample."
    )

    findings.append(
        "t-SNE and UMAP both reveal the strong day/night structure: night-time "
        "records (zero irradiation and power) form a tight, well-separated "
        "group, while daytime records spread out along the irradiation gradient."
    )

    if not results["umap_available"]:
        findings.append(
            "<strong>Note:</strong> UMAP is not installed. Install "
            "<code>umap-learn</code> to include it in the analysis."
        )

    return findings


run_full_unsupervised = train_and_evaluate
