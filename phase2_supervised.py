"""Phase 2: train and evaluate supervised heart-disease classifiers.

This script consumes the train/test matrices created by
``phase1_preprocess.py``, trains four binary classifiers, evaluates them on the
held-out test set, creates comparison plots, and saves all models and metrics.

Run:
    python phase2_supervised.py

Optional:
    python phase2_supervised.py --artifacts-dir artifacts
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
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Use a non-interactive backend so plots also work in terminals and CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)


LOGGER = logging.getLogger("phase2_supervised")
RANDOM_STATE: Final = 42
REQUIRED_DATA_KEYS: Final[set[str]] = {
    "X_train",
    "X_test",
    "y_train",
    "y_test",
}

MODEL_FILENAMES: Final[dict[str, str]] = {
    "Logistic Regression": "logistic_regression.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Support Vector Machine": "svm.joblib",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate four supervised heart-disease models."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Phase 1 input and Phase 2 output directory (default: artifacts).",
    )
    return parser.parse_args()


def load_preprocessed_data(
    artifacts_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the Phase 1 train/test artifact."""
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
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Feature matrices and targets contain different row counts.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Feature matrices contain missing or infinite values.")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("y_train must contain both binary classes 0 and 1.")
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("y_test must contain only binary classes 0 and 1.")

    return X_train, X_test, y_train, y_test


def build_models() -> dict[str, ClassifierMixin]:
    """Create reproducible baseline classifiers for model comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


def positive_class_scores(model: ClassifierMixin, features: np.ndarray) -> np.ndarray:
    """Return scores for class 1, preferring calibrated model probabilities."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        class_positions = np.flatnonzero(np.asarray(model.classes_) == 1)
        if class_positions.size != 1:
            raise ValueError("The fitted model does not expose positive class 1.")
        return np.asarray(probabilities[:, class_positions[0]], dtype=float)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)

    raise TypeError(f"{type(model).__name__} provides no ROC-compatible scores.")


def train_and_evaluate(
    models: dict[str, ClassifierMixin],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Fit each model and return metrics, predictions, and positive scores."""
    metric_rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}

    for name, model in models.items():
        LOGGER.info("Training %s...", name)
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=int)
        y_score = positive_class_scores(model, X_test)

        predictions[name] = y_pred
        scores[name] = y_score
        metric_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_score),
            }
        )

    metrics = pd.DataFrame(metric_rows).set_index("Model")
    return metrics, predictions, scores


def format_metrics_table(metrics: pd.DataFrame) -> str:
    """Format the comparison as a Markdown table without extra dependencies."""
    headers = ["Model", *metrics.columns.tolist()]
    rows = [
        [model_name, *(f"{value:.4f}" for value in row)]
        for model_name, row in metrics.iterrows()
    ]
    widths = [
        max(len(headers[index]), *(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(values: list[str]) -> str:
        cells = [str(value).ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def plot_confusion_matrices(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save a 2-by-2 grid of confusion matrices."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    for axis, (name, y_pred) in zip(axes.flat, predictions.items(), strict=True):
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(name, fontsize=12, fontweight="bold")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], labels=["Low Risk (0)", "High Risk (1)"])
        axis.set_yticks([0, 1], labels=["Low Risk (0)", "High Risk (1)"])

        threshold = matrix.max() / 2.0
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="white" if matrix[row_index, column_index] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Confusion Matrices on the Test Set", fontsize=16, fontweight="bold")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curves(
    y_test: np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save all model ROC curves on one graph."""
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)

    for name, y_score in scores.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_score)
        auc_value = roc_auc_score(y_test, y_score)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.2,
            label=f"{name} (AUC = {auc_value:.3f})",
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC = 0.500)")
    axis.set(
        title="ROC Curves on the Test Set",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_results(
    artifacts_dir: Path,
    models: dict[str, ClassifierMixin],
    metrics: pd.DataFrame,
) -> None:
    """Persist trained estimators and the evaluation summary."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, artifacts_dir / MODEL_FILENAMES[name])

    joblib.dump(metrics, artifacts_dir / "supervised_metrics.joblib")
    metrics.to_csv(artifacts_dir / "supervised_metrics.csv", index=True)

    # A single registry makes model loading in Streamlit straightforward.
    joblib.dump(models, artifacts_dir / "supervised_models.joblib")


def main() -> None:
    """Execute Phase 2 model training, evaluation, plotting, and persistence."""
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_preprocessed_data(artifacts_dir)
    models = build_models()
    metrics, predictions, scores = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    plot_confusion_matrices(
        y_test,
        predictions,
        artifacts_dir / "confusion_matrices.png",
    )
    plot_roc_curves(y_test, scores, artifacts_dir / "roc_curves.png")
    save_results(artifacts_dir, models, metrics)

    print("\nSupervised Model Performance (held-out test set)\n")
    print(format_metrics_table(metrics))
    best_model = metrics["ROC-AUC"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_model} ({metrics.loc[best_model, 'ROC-AUC']:.4f})")
    LOGGER.info("Saved Phase 2 models, metrics, and plots to %s.", artifacts_dir.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
