"""Phase 1: ingest and preprocess the Cleveland heart disease dataset.

The script filters ``heart.csv`` to the Cleveland cohort, binarizes the
diagnostic target, creates a stratified train/test split, and fits all learned
preprocessing steps on the training data only. Reusable artifacts are written
with joblib for the model-training and Streamlit phases.

Run:
    python phase1_preprocess.py

Optional:
    python phase1_preprocess.py --input heart.csv --output-dir artifacts
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger("phase1_preprocess")

RANDOM_STATE: Final = 42
TEST_SIZE: Final = 0.20
TARGET_SOURCE: Final = "num"
TARGET_NAME: Final = "target"

CONTINUOUS_COLUMNS: Final[list[str]] = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
]
CATEGORICAL_COLUMNS: Final[list[str]] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]
# ``ca`` is a discrete count. It is median-imputed but deliberately not scaled,
# matching the project's stated scaling requirements.
DISCRETE_NUMERIC_COLUMNS: Final[list[str]] = ["ca"]
FEATURE_COLUMNS: Final[list[str]] = (
    CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + DISCRETE_NUMERIC_COLUMNS
)
REQUIRED_COLUMNS: Final[set[str]] = {
    "id",
    "dataset",
    TARGET_SOURCE,
    *FEATURE_COLUMNS,
}
MISSING_TOKENS: Final[list[str]] = ["?", "", "NA", "N/A", "null", "None"]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Preprocess the UCI Cleveland heart disease cohort."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("heart.csv"),
        help="Path to the source CSV (default: heart.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for joblib artifacts (default: artifacts).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test-set fraction; must be between 0 and 1 (default: 0.20).",
    )
    return parser.parse_args()


def load_cleveland_data(csv_path: Path) -> pd.DataFrame:
    """Load the CSV, normalize missing tokens, and retain Cleveland rows."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path.resolve()}")

    dataframe = pd.read_csv(
        csv_path,
        na_values=MISSING_TOKENS,
        keep_default_na=True,
    )
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataset_names = dataframe["dataset"].astype("string").str.strip()
    cleveland = dataframe.loc[
        dataset_names.str.casefold().eq("cleveland")
    ].copy()
    if cleveland.empty:
        raise ValueError("No rows with dataset == 'Cleveland' were found.")

    cleveland.drop(columns=["id", "dataset"], inplace=True)
    return cleveland


def prepare_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Coerce feature types and convert ``num`` to a binary target."""
    target_numeric = pd.to_numeric(dataframe[TARGET_SOURCE], errors="coerce")
    if target_numeric.isna().any():
        bad_rows = target_numeric.index[target_numeric.isna()].tolist()
        raise ValueError(f"Target column contains invalid values at rows: {bad_rows}")
    if (target_numeric < 0).any():
        raise ValueError("Target column 'num' must contain non-negative values.")

    features = dataframe[FEATURE_COLUMNS].copy()
    for column in CONTINUOUS_COLUMNS + DISCRETE_NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        # scikit-learn's SimpleImputer expects np.nan here. Pandas' nullable
        # string dtype uses pd.NA, whose boolean comparison is ambiguous in
        # some pandas/scikit-learn combinations (notably on Python 3.14).
        normalized = features[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        features[column] = normalized.astype(object).where(
            normalized.notna(), np.nan
        )

    target = target_numeric.ge(1).astype("int8").rename(TARGET_NAME)
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build the train-fitted imputation, encoding, and scaling pipeline."""
    continuous_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    discrete_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("discrete", discrete_pipeline, DISCRETE_NUMERIC_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def missing_counts_after_imputation(
    fitted_preprocessor: ColumnTransformer,
    features: pd.DataFrame,
) -> pd.Series:
    """Report source-column missing counts after fitted imputers are applied."""
    counts: dict[str, int] = {}
    groups = (
        ("continuous", CONTINUOUS_COLUMNS),
        ("categorical", CATEGORICAL_COLUMNS),
        ("discrete", DISCRETE_NUMERIC_COLUMNS),
    )
    for transformer_name, columns in groups:
        imputer = fitted_preprocessor.named_transformers_[transformer_name].named_steps[
            "imputer"
        ]
        imputed = imputer.transform(features[columns])
        counts.update(
            {
                column: int(pd.isna(imputed[:, index]).sum())
                for index, column in enumerate(columns)
            }
        )
    return pd.Series(counts, dtype="int64").reindex(FEATURE_COLUMNS)


def save_artifacts(
    output_dir: Path,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Persist the reusable transformers, schema, and split datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = preprocessor.named_transformers_["continuous"].named_steps["scaler"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    schema = {
        "raw_feature_columns": FEATURE_COLUMNS,
        "continuous_columns": CONTINUOUS_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "discrete_numeric_columns": DISCRETE_NUMERIC_COLUMNS,
        "transformed_feature_names": feature_names,
        "target_name": TARGET_NAME,
        "target_mapping": {"num == 0": 0, "num >= 1": 1},
    }
    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.to_numpy(dtype=np.int8),
        "y_test": y_test.to_numpy(dtype=np.int8),
        "feature_names": feature_names,
    }

    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(encoder, output_dir / "categorical_encoder.joblib")
    joblib.dump(feature_names, output_dir / "feature_names.joblib")
    joblib.dump(schema, output_dir / "feature_schema.joblib")
    joblib.dump(split_data, output_dir / "preprocessed_data.joblib")


def print_summary(
    missing_before: pd.Series,
    missing_after: pd.Series,
    feature_count: int,
    target: pd.Series,
) -> None:
    """Print the required preprocessing diagnostics."""
    print("\nMissing values before imputation:")
    print(missing_before.to_string())
    print(f"Total missing before: {int(missing_before.sum())}")

    print("\nMissing values after imputation:")
    print(missing_after.to_string())
    print(f"Total missing after: {int(missing_after.sum())}")

    print(f"\nPreprocessed feature count: {feature_count}")
    distribution = pd.DataFrame(
        {
            "count": target.value_counts().sort_index(),
            "percentage": target.value_counts(normalize=True).sort_index().mul(100),
        }
    )
    distribution.index = distribution.index.map({0: "0 (absence)", 1: "1 (presence)"})
    print("\nBinary target class distribution:")
    print(distribution.to_string(formatters={"percentage": "{:.2f}%".format}))


def main() -> None:
    """Execute the complete Phase 1 preprocessing workflow."""
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be strictly between 0 and 1.")

    dataframe = load_cleveland_data(args.input)
    features, target = prepare_features_and_target(dataframe)
    missing_before = features.isna().sum().reindex(FEATURE_COLUMNS)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise RuntimeError("Preprocessed matrices unexpectedly contain missing values.")

    missing_after = missing_counts_after_imputation(preprocessor, features)
    save_artifacts(
        args.output_dir,
        preprocessor,
        feature_names,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    print_summary(missing_before, missing_after, len(feature_names), target)
    LOGGER.info(
        "Saved Phase 1 artifacts to %s (train=%d, test=%d).",
        args.output_dir.resolve(),
        len(y_train),
        len(y_test),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
