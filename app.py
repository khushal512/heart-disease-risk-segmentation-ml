"""Streamlit dashboard for heart-disease prediction and segmentation.

Run from the project directory with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Heart Disease Risk & Segmentation Dashboard",
    layout="wide",
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

MODEL_FILES = {
    "Support Vector Machine": "svm.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Decision Tree": "decision_tree.joblib",
}
MODEL_DISPLAY_NAMES = {
    "Support Vector Machine": "Support Vector Machine (SVM)",
    "Logistic Regression": "Logistic Regression",
    "K-Nearest Neighbours": "K-Nearest Neighbours (KNN)",
    "Decision Tree": "Decision Tree",
}
MODEL_ROLES = {
    "Support Vector Machine": "Top performer by ROC-AUC",
    "Logistic Regression": "Interpretable linear baseline",
    "K-Nearest Neighbours": "Similarity-based classifier",
    "Decision Tree": "Rule-based classifier",
}

RAW_FEATURE_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
    "ca",
]


st.markdown(
    """
    <style>
      .main-title {font-size: 2.15rem; font-weight: 750; margin-bottom: 0.15rem;}
      .subtle {color: #64748b; margin-bottom: 1.25rem;}
      .result-card {
        border-radius: 0.8rem; padding: 1.15rem 1.3rem; margin: 0.75rem 0;
        border: 1px solid rgba(148, 163, 184, 0.35);
      }
      .risk-high {background: rgba(220, 38, 38, 0.12); border-left: 0.5rem solid #dc2626;}
      .risk-low {background: rgba(22, 163, 74, 0.12); border-left: 0.5rem solid #16a34a;}
      .risk-split {background: rgba(245, 158, 11, 0.14); border-left: 0.5rem solid #f59e0b;}
      .card-title {font-size: 1.3rem; font-weight: 750; margin-bottom: 0.2rem;}
      .model-card {
        border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 0.75rem;
        padding: 0.9rem 1rem; min-height: 7rem; margin-bottom: 0.55rem;
      }
      .model-name {font-weight: 750; font-size: 1.02rem; margin-bottom: 0.6rem;}
      .badge-high, .badge-low {
        display: inline-block; border-radius: 999px; padding: 0.2rem 0.65rem;
        font-weight: 700; font-size: 0.88rem;
      }
      .badge-high {background: rgba(220, 38, 38, 0.16); color: #dc2626;}
      .badge-low {background: rgba(22, 163, 74, 0.16); color: #16a34a;}
      .metric-card {
        background: rgba(37, 99, 235, 0.08); border-left: 0.4rem solid #2563eb;
        border-radius: 0.7rem; padding: 1rem 1.2rem; margin: 0.6rem 0 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def artifact_path(filename: str) -> Path:
    """Resolve an artifact path relative to this application."""
    return ARTIFACTS_DIR / filename


@st.cache_resource(show_spinner=False)
def load_joblib_resource(path_string: str, modified_time: float) -> Any:
    """Load a model-like joblib artifact and cache by path and modification time."""
    del modified_time
    return joblib.load(path_string)


@st.cache_data(show_spinner=False)
def load_joblib_data(path_string: str, modified_time: float) -> Any:
    """Load a data-like joblib artifact and cache by path and modification time."""
    del modified_time
    return joblib.load(path_string)


def safe_load(filename: str, *, resource: bool = False) -> Any | None:
    """Load one artifact, displaying an actionable warning on failure."""
    path = artifact_path(filename)
    if not path.is_file():
        st.error(f"Missing file: {path}")
        return None
    try:
        loader = load_joblib_resource if resource else load_joblib_data
        return loader(str(path), path.stat().st_mtime)
    except Exception as error:  # Streamlit should remain usable if one artifact is bad.
        st.error(f"Could not load `{path.name}`: {error}")
        return None


def load_models() -> dict[str, Any] | None:
    """Load the combined registry, falling back to individual model files."""
    registry_path = artifact_path("supervised_models.joblib")
    if registry_path.is_file():
        registry = safe_load("supervised_models.joblib", resource=True)
        if isinstance(registry, dict) and all(
            name in registry for name in MODEL_FILES
        ):
            return registry

    models: dict[str, Any] = {}
    for model_name, filename in MODEL_FILES.items():
        model = safe_load(filename, resource=True)
        if model is None:
            return None
        models[model_name] = model
    return models


def build_patient_dataframe(values: dict[str, Any]) -> pd.DataFrame:
    """Create a schema-ordered one-row DataFrame for the saved preprocessor."""
    missing = set(RAW_FEATURE_COLUMNS).difference(values)
    if missing:
        raise ValueError(f"Patient input is missing: {', '.join(sorted(missing))}")
    return pd.DataFrame([{column: values[column] for column in RAW_FEATURE_COLUMNS}])


def positive_probability(model: Any, features: np.ndarray) -> float:
    """Return the fitted model's probability for diagnostic class 1."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = np.asarray(model.classes_)
        positions = np.flatnonzero(classes == 1)
        if positions.size != 1:
            raise ValueError("Selected model does not expose diagnostic class 1.")
        return float(probabilities[0, positions[0]])
    if hasattr(model, "decision_function"):
        score = float(np.asarray(model.decision_function(features)).reshape(-1)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    raise TypeError("Selected model does not provide probability-compatible scores.")


def predict_with_model(
    model_name: str, model: Any, features: np.ndarray
) -> dict[str, Any]:
    """Generate one model's class, disease probability, and verdict confidence."""
    prediction = int(model.predict(features)[0])
    disease_probability = positive_probability(model, features)
    confidence = disease_probability if prediction == 1 else 1.0 - disease_probability
    return {
        "model_key": model_name,
        "model_name": MODEL_DISPLAY_NAMES[model_name],
        "prediction": prediction,
        "diagnosis": "High Risk" if prediction == 1 else "Low Risk / Absence",
        "disease_probability": disease_probability,
        "confidence": confidence,
        "role": MODEL_ROLES[model_name],
    }


def run_all_models(
    models: dict[str, Any], features: np.ndarray
) -> list[dict[str, Any]]:
    """Evaluate all four estimators in one request and preserve display order.

    The estimators are intentionally invoked as a deterministic batch instead
    of Python threads. Some scikit-learn native estimators can deadlock under
    threaded inference on Windows/Python 3.14; single-row inference remains
    effectively instantaneous while all results are still presented together.
    """
    missing_models = set(MODEL_FILES).difference(models)
    if missing_models:
        raise KeyError("Missing trained models: " + ", ".join(sorted(missing_models)))
    return [
        predict_with_model(name, models[name], features) for name in MODEL_FILES
    ]


def show_consensus_panel(results: list[dict[str, Any]]) -> None:
    """Render consensus, four model cards, and a comparison table."""
    high_risk_count = sum(result["prediction"] for result in results)
    if high_risk_count == len(results):
        card_class = "risk-high"
        banner = "Strong Consensus: 4 of 4 models predict High Risk"
    elif high_risk_count == 0:
        card_class = "risk-low"
        banner = "Strong Consensus: 4 of 4 models predict Low Risk"
    else:
        card_class = "risk-split"
        banner = f"Split Decision: {high_risk_count} of 4 models predict High Risk"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
          <div class="card-title">{banner}</div>
          <div>Review the individual model estimates below. Mixed results indicate
          model uncertainty and should not be treated as a clinical diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    for column, result in zip(columns, results, strict=True):
        badge_class = "badge-high" if result["prediction"] == 1 else "badge-low"
        with column:
            st.markdown(
                f"""
                <div class="model-card">
                  <div class="model-name">{result['model_name']}</div>
                  <span class="{badge_class}">{result['diagnosis']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Verdict confidence: {result['confidence']:.1%}")
            st.progress(
                min(max(int(round(result["confidence"] * 100)), 0), 100),
                text=f"{result['confidence']:.1%}",
            )

    summary = pd.DataFrame(
        [
            {
                "Model Name": result["model_name"],
                "Predicted Diagnosis": result["diagnosis"],
                "Confidence Probability (%)": result["confidence"] * 100,
                "Model Role": result["role"],
            }
            for result in results
        ]
    )
    st.markdown("#### Model Comparison Summary")
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "Confidence Probability (%)": st.column_config.NumberColumn(
                "Confidence Probability (%)", format="%.1f%%"
            )
        },
    )
    st.caption(
        "Confidence is the probability assigned to each model's displayed verdict. "
        "This dashboard is for educational use and is not a medical diagnosis."
    )


def render_cluster_plot(
    clustered_data: pd.DataFrame,
    patient_projection: np.ndarray | None,
    patient_cluster: int | None,
) -> None:
    """Render the cohort PCA projection and optional submitted-patient marker."""
    plot_data = clustered_data[["pca_1", "pca_2", "kmeans_cluster", "target"]].copy()
    plot_data["Cluster"] = plot_data["kmeans_cluster"].map(
        lambda value: f"Cluster {int(value)}"
    )
    plot_data["Diagnosis"] = plot_data["target"].map(
        {0: "No disease", 1: "Heart disease"}
    )

    base_chart = (
        alt.Chart(plot_data)
        .mark_circle(size=75, opacity=0.72, stroke="white", strokeWidth=0.4)
        .encode(
            x=alt.X("pca_1:Q", title="Principal Component 1"),
            y=alt.Y("pca_2:Q", title="Principal Component 2"),
            color=alt.Color(
                "Cluster:N",
                scale=alt.Scale(range=["#2563eb", "#f59e0b"]),
                legend=alt.Legend(title="K-Means Segment"),
            ),
            tooltip=[
                alt.Tooltip("Cluster:N"),
                alt.Tooltip("Diagnosis:N"),
                alt.Tooltip("pca_1:Q", title="PC1", format=".3f"),
                alt.Tooltip("pca_2:Q", title="PC2", format=".3f"),
            ],
        )
    )

    chart: alt.Chart | alt.LayerChart = base_chart
    if patient_projection is not None:
        patient_data = pd.DataFrame(
            {
                "pca_1": [float(patient_projection[0])],
                "pca_2": [float(patient_projection[1])],
                "Patient": [f"Current patient (Cluster {patient_cluster})"],
            }
        )
        patient_chart = (
            alt.Chart(patient_data)
            .mark_point(
                shape="diamond",
                size=500,
                filled=True,
                color="#dc2626",
                stroke="white",
                strokeWidth=1.8,
            )
            .encode(
                x="pca_1:Q",
                y="pca_2:Q",
                tooltip=[
                    alt.Tooltip("Patient:N"),
                    alt.Tooltip("pca_1:Q", title="PC1", format=".3f"),
                    alt.Tooltip("pca_2:Q", title="PC2", format=".3f"),
                ],
            )
        )
        chart = alt.layer(base_chart, patient_chart)

    chart = (
        chart.properties(
            title="Cleveland Cohort in Two-Dimensional PCA Space",
            height=560,
        )
        .interactive()
        .configure_axis(gridOpacity=0.18)
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, width="stretch")


def show_saved_image(filename: str, caption: str) -> None:
    """Display a saved image or an actionable warning if it is unavailable."""
    path = artifact_path(filename)
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.error(f"Missing file: {path}")


st.markdown(
    '<div class="main-title">Heart Disease Risk Prediction and Patient Segmentation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtle">Supervised risk estimation and unsupervised clinical cohort analysis</div>',
    unsafe_allow_html=True,
)

if "patient_transformed" not in st.session_state:
    st.session_state.patient_transformed = None
if "patient_raw" not in st.session_state:
    st.session_state.patient_raw = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

diagnosis_tab, segmentation_tab, performance_tab = st.tabs(
    [
        "Patient Diagnosis",
        "Patient Segmentation",
        "Model Performance & Lab Evaluation",
    ]
)


with diagnosis_tab:
    st.subheader("Patient Diagnosis (Supervised Classification)")
    st.write(
        "Enter the patient's clinical measurements to compare all four trained models. "
        "Inputs are transformed by the exact preprocessing pipeline fitted in Phase 1."
    )

    with st.form("patient_diagnosis_form"):
        first, second, third = st.columns(3)
        with first:
            age = st.number_input(
                "Age (years)",
                18,
                100,
                54,
                1,
                help="Patient's age in years. Heart attack risk naturally climbs as blood vessels stiffen and age over time.",
            )
            sex = st.selectbox(
                "Sex",
                ["Male", "Female"],
                help="Biological sex (Male / Female). Men generally face heart disease earlier in life, while women's risk increases later.",
            )
            cp = st.selectbox(
                "Chest Pain Type",
                [
                    "typical angina",
                    "atypical angina",
                    "non-anginal",
                    "asymptomatic",
                ],
                index=3,
                help="How chest pain feels: Typical Angina (tight squeezing pressure during exertion), Atypical (unusual pain patterns), Non-Anginal (unrelated pain like acid reflux or muscle pull), or Asymptomatic (no pain felt at all).",
            )
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                70,
                250,
                130,
                1,
                help="Resting blood pressure in mm Hg. Around 120 is healthy; levels above 140 put dangerous, constant strain on your artery walls.",
            )
            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                80,
                700,
                245,
                1,
                help="Total circulating blood cholesterol in mg/dl. Under 200 is healthy; 240 or higher builds sticky fatty plaque that clogs arteries.",
            )

        with second:
            fbs_label = st.radio(
                "Fasting Blood Sugar > 120 mg/dl",
                ["No", "Yes"],
                horizontal=True,
                help="Is morning fasting sugar above 120 mg/dl? (Yes / No). Indicates diabetes or pre-diabetes, which silently damages cardiovascular vessels.",
            )
            restecg = st.selectbox(
                "Resting ECG",
                ["normal", "st-t abnormality", "lv hypertrophy"],
                help="Resting electrical heartbeat trace: Normal (steady rhythm), ST-T Abnormality (mild heart strain), or LV Hypertrophy (heart pumping muscle is overworked and thickened).",
            )
            thalch = st.number_input(
                "Maximum Heart Rate Achieved",
                60,
                250,
                150,
                1,
                help="Top speed the heart reached during treadmill exercise (bpm). A healthy heart reaches 150–180+ bpm; a damaged heart tires out much earlier.",
            )
            exang_label = st.radio(
                "Exercise-Induced Angina",
                ["No", "Yes"],
                horizontal=True,
                help="Did chest pain start during the treadmill workout? (Yes / No). If yes, the heart muscle is starving for oxygen under physical demand.",
            )
            oldpeak = st.slider(
                "ST Depression Induced by Exercise",
                0.0,
                7.0,
                1.0,
                0.1,
                help="How deeply the ECG wave dips under exercise stress (in mm). 0 mm is healthy; dips of 1.5 to 3+ mm mean the heart muscle struggled for oxygen.",
            )

        with third:
            slope = st.selectbox(
                "Slope of Peak Exercise ST",
                ["upsloping", "flat", "downsloping"],
                index=1,
                help="Direction of the ECG wave at peak exertion: Upsloping (normal, healthy recovery), Flat (moderate heart strain), or Downsloping (severe lack of blood flow).",
            )
            ca = st.selectbox(
                "Major Vessels Colored by Fluoroscopy",
                [0, 1, 2, 3],
                help="Number of major heart blood vessels (0 to 3) showing blockage on an X-ray dye scan. 0 means completely clear; 1 to 3 means significant blockages exist.",
            )
            thal = st.selectbox(
                "Thalassemia",
                ["normal", "fixed defect", "reversable defect"],
                help="How well blood reaches heart muscle: Normal (full blood flow), Fixed Defect (permanent scar tissue from an old heart attack), or Reversible Defect (blood flow drops during exercise but recovers at rest).",
            )

        submitted = st.form_submit_button(
            "Predict Heart Disease Risk", type="primary", width="stretch"
        )

    if submitted:
        # Load model artifacts only after the form is fully rendered and the
        # user requests a prediction. Missing artifacts therefore never hide
        # the clinical controls or prevent the remaining tabs from rendering.
        preprocessor = safe_load("preprocessor.joblib", resource=True)
        models = load_models()
        if preprocessor is None or models is None:
            st.error(
                "Prediction is unavailable because one or more required model "
                "artifacts could not be loaded."
            )
        else:
            patient_values = {
                "age": age,
                "trestbps": trestbps,
                "chol": chol,
                "thalch": thalch,
                "oldpeak": oldpeak,
                "sex": sex,
                "cp": cp,
                "fbs": fbs_label == "Yes",
                "restecg": restecg,
                "exang": exang_label == "Yes",
                "slope": slope,
                "thal": thal,
                "ca": ca,
            }
            try:
                raw_patient = build_patient_dataframe(patient_values)
                transformed_patient = np.asarray(
                    preprocessor.transform(raw_patient), dtype=float
                )
                prediction_results = run_all_models(models, transformed_patient)
                st.session_state.patient_raw = patient_values
                st.session_state.patient_transformed = transformed_patient
                st.session_state.prediction_result = prediction_results
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    if st.session_state.prediction_result is not None:
        show_consensus_panel(st.session_state.prediction_result)


with segmentation_tab:
    st.subheader("Patient Segmentation (Unsupervised Analysis)")

    clustered_data = safe_load("clustered_data.joblib")
    cluster_summaries = safe_load("cluster_outcome_summaries.joblib")
    kmeans_model = safe_load("kmeans_model.joblib", resource=True)
    pca_transformer = safe_load("pca_transformer.joblib", resource=True)

    if isinstance(clustered_data, pd.DataFrame):
        required_cluster_columns = {
            "kmeans_cluster",
            "target",
            "pca_1",
            "pca_2",
        }
        if not required_cluster_columns.issubset(clustered_data.columns):
            st.error("The clustered-data artifact does not contain the required columns.")
        else:
            cluster_table = pd.crosstab(
                clustered_data["kmeans_cluster"], clustered_data["target"]
            ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            cluster_table.columns = ["No Disease (0)", "Heart Disease (1)"]
            cluster_table.index.name = "K-Means Cluster"

            disease_rates = (
                clustered_data.groupby("kmeans_cluster")["target"].mean().mul(100)
            )
            disease_counts = clustered_data.groupby("kmeans_cluster")["target"].sum()
            high_risk_cluster = int(disease_rates.idxmax())
            disease_prevalence = float(disease_rates.loc[high_risk_cluster])
            disease_capture = float(
                disease_counts.loc[high_risk_cluster]
                / clustered_data["target"].sum()
                * 100
            )

            left_metric, middle_metric, right_metric = st.columns(3)
            left_metric.metric("Higher-Risk Segment", f"Cluster {high_risk_cluster}")
            middle_metric.metric(
                "Disease Prevalence", f"{disease_prevalence:.2f}%"
            )
            right_metric.metric(
                "Diagnosed Cases Captured", f"{disease_capture:.2f}%"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                  <b>K-Means insight:</b> Cluster {high_risk_cluster} has
                  {disease_prevalence:.2f}% heart-disease prevalence and contains
                  {disease_capture:.2f}% of all diagnosed patients in the cohort.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Cluster vs Diagnostic Outcome")
            st.dataframe(cluster_table, width="stretch")

            if isinstance(cluster_summaries, dict) and "K-Means" in cluster_summaries:
                with st.expander("Detailed disease-capture statistics"):
                    st.dataframe(
                        cluster_summaries["K-Means"][1].style.format(
                            {
                                "Disease Rate Within Cluster (%)": "{:.2f}%",
                                "Share of All Disease Cases (%)": "{:.2f}%",
                            }
                        ),
                        width="stretch",
                    )

            patient_projection = None
            patient_cluster = None
            if st.session_state.patient_transformed is not None:
                if pca_transformer is not None and kmeans_model is not None:
                    try:
                        patient_projection = pca_transformer.transform(
                            st.session_state.patient_transformed
                        )[0]
                        patient_cluster = int(
                            kmeans_model.predict(
                                st.session_state.patient_transformed
                            )[0]
                        )
                        st.info(
                            f"The patient submitted in Tab 1 belongs to K-Means "
                            f"Cluster {patient_cluster}."
                        )
                    except Exception as error:
                        st.warning(f"Could not project the current patient: {error}")
            else:
                st.info(
                    "Submit a patient in Tab 1 to overlay that record on the PCA plot."
                )

            st.markdown("#### Interactive PCA Cluster Projection")
            render_cluster_plot(
                clustered_data, patient_projection, patient_cluster
            )
    else:
        st.info("Run Phase 3 to enable segmentation analysis.")


with performance_tab:
    st.subheader("Model Performance & Lab Evaluation")

    metrics = safe_load("supervised_metrics.joblib")
    if metrics is None:
        metrics_csv = artifact_path("supervised_metrics.csv")
        if metrics_csv.is_file():
            try:
                metrics = pd.read_csv(metrics_csv, index_col=0)
            except Exception as error:
                st.error(f"Could not load `{metrics_csv.name}`: {error}")

    if isinstance(metrics, pd.DataFrame):
        st.markdown("#### Held-Out Test-Set Metrics")
        st.dataframe(
            metrics.style.format("{:.4f}").highlight_max(
                axis=0, color="rgba(22, 163, 74, 0.20)"
            ),
            width="stretch",
        )

        if "Support Vector Machine" in metrics.index:
            svm_accuracy = float(metrics.loc["Support Vector Machine", "Accuracy"])
            svm_auc = float(metrics.loc["Support Vector Machine", "ROC-AUC"])
            st.markdown(
                f"""
                <div class="metric-card">
                  <b>Top-performing model:</b> Support Vector Machine achieved
                  {svm_accuracy:.2%} accuracy and {svm_auc:.4f} ROC-AUC on the
                  held-out Cleveland test set.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("Supervised metrics are unavailable. Re-run Phase 2.")

    st.markdown("#### Confusion Matrices")
    show_saved_image(
        "confusion_matrices.png",
        "Confusion matrices for all four supervised classifiers",
    )

    st.markdown("#### Receiver Operating Characteristic Curves")
    show_saved_image(
        "roc_curves.png",
        "Combined ROC curves and area-under-curve values",
    )

    st.markdown("#### Hierarchical Clustering Dendrogram")
    show_saved_image(
        "hierarchical_dendrogram.png",
        "Ward-linkage hierarchical clustering dendrogram",
    )

st.divider()
st.caption(
    "Educational machine-learning demonstration using the UCI Cleveland heart "
    "disease cohort. Outputs must not be used as a clinical diagnosis."
)
