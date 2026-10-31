"""Phase 3: unsupervised patient segmentation for the Cleveland cohort.

The script combines the Phase 1 train/test feature matrices, evaluates K-Means
solutions for k=2 through k=10, fits two-cluster K-Means and Ward hierarchical
models, compares their clusters with the diagnostic target, creates PCA and
dendrogram visualizations, and persists reusable artifacts.

Run:
    python phase3_unsupervised.py

Optional:
    python phase3_unsupervised.py --artifacts-dir artifacts
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (select backend before pyplot)


LOGGER = logging.getLogger("phase3_unsupervised")

RANDOM_STATE: Final = 42
N_CLUSTERS: Final = 2
K_RANGE: Final[range] = range(2, 11)
MAX_DENDROGRAM_SAMPLES: Final = 150
REQUIRED_DATA_KEYS: Final[set[str]] = {
    "X_train",
    "X_test",
    "y_train",
    "y_test",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run unsupervised patient segmentation for the Cleveland cohort."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Phase 1 input and Phase 3 output directory (default: artifacts).",
    )
    return parser.parse_args()


def load_full_cohort(
    artifacts_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Combine Phase 1 train/test matrices without adding target to features."""
    data_path = artifacts_dir / "preprocessed_data.joblib"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Phase 1 artifact not found: {data_path.resolve()}. "
            "Run phase1_preprocess.py first."
        )

    data: Any = joblib.load(data_path)
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a mapping in {data_path}, got {type(data).__name__}.")

    missing_keys = REQUIRED_DATA_KEYS.difference(data)
    if missing_keys:
        raise KeyError(
            "Preprocessed data is missing keys: " + ", ".join(sorted(missing_keys))
        )

    X_train = np.asarray(data["X_train"], dtype=float)
    X_test = np.asarray(data["X_test"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=int).reshape(-1)
    y_test = np.asarray(data["y_test"], dtype=int).reshape(-1)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be two-dimensional matrices.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test matrices have different feature counts.")
    if X_train.shape[0] != y_train.size or X_test.shape[0] != y_test.size:
        raise ValueError("Feature matrices and target arrays have different row counts.")

    features = np.vstack((X_train, X_test))
    target = np.concatenate((y_train, y_test))
    if not np.isfinite(features).all():
        raise ValueError("The combined feature matrix contains NaN or infinite values.")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("The combined target must contain binary classes 0 and 1.")

    raw_names = data.get("feature_names")
    if raw_names is None:
        names_path = artifacts_dir / "feature_names.joblib"
        if not names_path.is_file():
            raise FileNotFoundError("No transformed feature-name artifact was found.")
        raw_names = joblib.load(names_path)
    feature_names = [str(name) for name in raw_names]
    if len(feature_names) != features.shape[1]:
        raise ValueError(
            "Feature-name count does not match the transformed matrix width: "
            f"{len(feature_names)} != {features.shape[1]}."
        )

    return features, target, feature_names


def evaluate_kmeans_candidates(features: np.ndarray) -> pd.DataFrame:
    """Calculate inertia and silhouette score for k=2 through k=10."""
    rows: list[dict[str, float | int]] = []
    for cluster_count in K_RANGE:
        candidate = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = candidate.fit_predict(features)
        rows.append(
            {
                "k": cluster_count,
                "inertia": float(candidate.inertia_),
                "silhouette_score": float(silhouette_score(features, labels)),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def fit_cluster_models(
    features: np.ndarray,
) -> tuple[KMeans, AgglomerativeClustering, np.ndarray, np.ndarray]:
    """Fit the required two-cluster K-Means and Ward models."""
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    agglomerative = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        linkage="ward",
    )
    kmeans_labels = kmeans.fit_predict(features)
    agglomerative_labels = agglomerative.fit_predict(features)
    return kmeans, agglomerative, kmeans_labels, agglomerative_labels


def cluster_outcome_summary(
    labels: np.ndarray,
    target: np.ndarray,
    cluster_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a contingency table and clinically interpretable percentages."""
    cluster_series = pd.Series(labels, name=cluster_name)
    target_series = pd.Series(target, name="target")
    contingency = pd.crosstab(cluster_series, target_series).reindex(
        index=range(N_CLUSTERS), columns=[0, 1], fill_value=0
    )
    contingency.columns = ["No Disease (0)", "Heart Disease (1)"]

    cluster_totals = contingency.sum(axis=1)
    disease_counts = contingency["Heart Disease (1)"]
    total_disease = int(disease_counts.sum())
    percentages = pd.DataFrame(
        {
            "Cluster Size": cluster_totals,
            "Heart Disease Cases": disease_counts,
            "Disease Rate Within Cluster (%)": (
                disease_counts.div(cluster_totals).mul(100)
            ),
            "Share of All Disease Cases (%)": (
                disease_counts.div(total_disease).mul(100)
            ),
        }
    )
    percentages.index.name = cluster_name
    return contingency, percentages


def print_analysis(
    diagnostics: pd.DataFrame,
    target: np.ndarray,
    summaries: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Print K diagnostics and cluster/outcome distributions."""
    print("\nK-Means cluster-count diagnostics (k=2..10)\n")
    printable = diagnostics.rename(
        columns={"inertia": "Inertia", "silhouette_score": "Silhouette Score"}
    )
    print(
        printable.to_string(
            formatters={
                "Inertia": "{:.3f}".format,
                "Silhouette Score": "{:.4f}".format,
            }
        )
    )
    best_k = int(diagnostics["silhouette_score"].idxmax())
    print(
        f"\nBest candidate by silhouette score: k={best_k} "
        f"({diagnostics.loc[best_k, 'silhouette_score']:.4f})"
    )
    print(
        f"Required final segmentation: k={N_CLUSTERS} "
        "for comparison with the binary diagnostic outcome."
    )
    print(
        f"\nFull cohort: {target.size} patients; "
        f"heart disease present in {int(target.sum())} "
        f"({target.mean() * 100:.2f}%)."
    )

    for title, (contingency, percentages) in summaries.items():
        print(f"\n{title}: cluster vs diagnostic outcome\n")
        print(contingency.to_string())
        print(f"\n{title}: disease capture summary\n")
        print(
            percentages.to_string(
                formatters={
                    "Disease Rate Within Cluster (%)": "{:.2f}%".format,
                    "Share of All Disease Cases (%)": "{:.2f}%".format,
                }
            )
        )


def fit_pca(features: np.ndarray) -> tuple[PCA, np.ndarray]:
    """Fit a reusable two-component PCA projection."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(features)
    return pca, coordinates


def plot_pca_comparison(
    coordinates: np.ndarray,
    kmeans_labels: np.ndarray,
    agglomerative_labels: np.ndarray,
    target: np.ndarray,
    explained_variance: np.ndarray,
    output_path: Path,
) -> None:
    """Plot both cluster solutions and ground truth in a shared PCA space."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    panels = (
        ("K-Means Clusters", kmeans_labels, "viridis"),
        ("Agglomerative Clusters", agglomerative_labels, "plasma"),
        ("Ground-Truth Diagnosis", target, "coolwarm"),
    )

    for axis, (title, colors, colormap) in zip(axes, panels, strict=True):
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=colors,
            cmap=colormap,
            s=38,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% variance)")
        axis.grid(alpha=0.2)
        colorbar = figure.colorbar(scatter, ax=axis, ticks=[0, 1], fraction=0.046)
        colorbar.set_label("Cluster" if "Clusters" in title else "Target")

    figure.suptitle(
        "Patient Segmentation and Diagnostic Outcome in PCA Space",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_sample_dendrogram(features: np.ndarray, output_path: Path) -> np.ndarray:
    """Create a reproducible Ward dendrogram from at most 150 patients."""
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(MAX_DENDROGRAM_SAMPLES, features.shape[0])
    sample_indices = np.sort(
        rng.choice(features.shape[0], size=sample_size, replace=False)
    )
    linkage_matrix = linkage(features[sample_indices], method="ward")

    figure, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    dendrogram(
        linkage_matrix,
        ax=axis,
        truncate_mode="lastp",
        p=35,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=9,
        color_threshold=None,
    )
    axis.set_title(
        f"Ward Hierarchical Clustering Dendrogram (sample n={sample_size})",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Merged patient group")
    axis.set_ylabel("Ward distance")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return sample_indices


def build_clustered_dataframe(
    features: np.ndarray,
    feature_names: list[str],
    target: np.ndarray,
    kmeans_labels: np.ndarray,
    agglomerative_labels: np.ndarray,
    coordinates: np.ndarray,
) -> pd.DataFrame:
    """Combine transformed features, labels, target, and PCA coordinates."""
    dataframe = pd.DataFrame(features, columns=feature_names)
    dataframe["kmeans_cluster"] = kmeans_labels
    dataframe["agglomerative_cluster"] = agglomerative_labels
    dataframe["target"] = target
    dataframe["pca_1"] = coordinates[:, 0]
    dataframe["pca_2"] = coordinates[:, 1]
    return dataframe


def save_artifacts(
    artifacts_dir: Path,
    kmeans: KMeans,
    agglomerative: AgglomerativeClustering,
    pca: PCA,
    clustered_data: pd.DataFrame,
    diagnostics: pd.DataFrame,
    summaries: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    dendrogram_sample_indices: np.ndarray,
) -> None:
    """Persist models and analysis outputs for later phases."""
    joblib.dump(kmeans, artifacts_dir / "kmeans_model.joblib")
    joblib.dump(agglomerative, artifacts_dir / "agglomerative_model.joblib")
    joblib.dump(pca, artifacts_dir / "pca_transformer.joblib")
    joblib.dump(clustered_data, artifacts_dir / "clustered_data.joblib")
    joblib.dump(diagnostics, artifacts_dir / "kmeans_diagnostics.joblib")
    diagnostics.to_csv(artifacts_dir / "kmeans_diagnostics.csv")
    joblib.dump(summaries, artifacts_dir / "cluster_outcome_summaries.joblib")
    joblib.dump(
        dendrogram_sample_indices,
        artifacts_dir / "dendrogram_sample_indices.joblib",
    )


def main() -> None:
    """Execute Phase 3 segmentation, analysis, plotting, and persistence."""
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    features, target, feature_names = load_full_cohort(artifacts_dir)
    diagnostics = evaluate_kmeans_candidates(features)
    kmeans, agglomerative, kmeans_labels, agglomerative_labels = (
        fit_cluster_models(features)
    )

    summaries = {
        "K-Means": cluster_outcome_summary(
            kmeans_labels, target, "kmeans_cluster"
        ),
        "Agglomerative": cluster_outcome_summary(
            agglomerative_labels, target, "agglomerative_cluster"
        ),
    }
    print_analysis(diagnostics, target, summaries)

    pca, coordinates = fit_pca(features)
    plot_pca_comparison(
        coordinates,
        kmeans_labels,
        agglomerative_labels,
        target,
        pca.explained_variance_ratio_,
        artifacts_dir / "clustering_pca_plots.png",
    )
    sample_indices = plot_sample_dendrogram(
        features,
        artifacts_dir / "hierarchical_dendrogram.png",
    )
    clustered_data = build_clustered_dataframe(
        features,
        feature_names,
        target,
        kmeans_labels,
        agglomerative_labels,
        coordinates,
    )
    save_artifacts(
        artifacts_dir,
        kmeans,
        agglomerative,
        pca,
        clustered_data,
        diagnostics,
        summaries,
        sample_indices,
    )

    LOGGER.info(
        "Saved Phase 3 models, analyses, and plots to %s.",
        artifacts_dir.resolve(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
