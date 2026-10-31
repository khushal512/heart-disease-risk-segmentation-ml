# Heart Disease Risk Prediction & Patient Segmentation

An end-to-end Machine Learning pipeline and interactive clinical dashboard combining supervised classification and unsupervised clustering to predict heart disease risk and discover patient phenotypes. Developed for **CSE 3148: Machine Learning Mini-Project (2026)** at **Manipal Institute of Technology, Manipal**.

---

## 📌 Project Overview

Cardiovascular disease is a leading cause of mortality globally. This project addresses risk stratification through two complementary approaches:
1. **Supervised Classification:** Compares four classical classifiers (Logistic Regression, K-Nearest Neighbours, Decision Tree, and Support Vector Machine) to predict coronary heart disease presence based on physiological measurements[cite: 1].
2. **Unsupervised Patient Segmentation:** Implements K-Means and Agglomerative Hierarchical Clustering (without diagnostic labels) to reveal natural clinical groupings and correlates them with real diagnostic outcomes[cite: 1].
3. **Interactive Clinical Dashboard:** A Streamlit web application providing simultaneous 4-model inference, plain-language clinical tooltips, and real-time 2D PCA patient cluster mapping.

---

## 📊 Dataset

* **Source:** [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland Clinic Foundation collection)[cite: 1]
* **Cohort Size:** 304 patient records[cite: 1]
* **Target:** Binary diagnosis (`0` = Absence, `1` = Presence of coronary artery disease with $\ge 50\%$ vessel diameter narrowing)[cite: 1]
* **Input Features (13 Clinical Attributes):**
  * `age`: Patient age in years[cite: 1]
  * `sex`: Biological sex (Male/Female)[cite: 1]
  * `cp`: Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic)[cite: 1]
  * `trestbps`: Resting blood pressure in mm Hg[cite: 1]
  * `chol`: Serum cholesterol in mg/dl[cite: 1]
  * `fbs`: Fasting blood sugar > 120 mg/dl[cite: 1]
  * `restecg`: Resting electrocardiographic results[cite: 1]
  * `thalach`: Maximum heart rate achieved during exercise[cite: 1]
  * `exang`: Exercise-induced angina[cite: 1]
  * `oldpeak`: ST depression induced by exercise relative to rest[cite: 1]
  * `slope`: Slope of peak exercise ST segment[cite: 1]
  * `ca`: Number of major vessels (0–3) colored by fluoroscopy
  * `thal`: Thallium scintigraphy stress defect status[cite: 1]

---

## 📈 Experimental Results

### 1. Supervised Classification Performance (80:20 Stratified Holdout Split)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **0.9016** | **0.8667** | **0.9286** | **0.8966** | **0.9627** |
| **Logistic Regression** | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9610 |
| **K-Nearest Neighbours (KNN)** | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9437 |
| **Decision Tree** | 0.6721 | 0.6176 | 0.7500 | 0.6774 | 0.6780 |

* **Best Supervised Model:** Support Vector Machine achieved the highest diagnostic discriminability with an **ROC-AUC of 0.9627** and a clinical sensitivity (**Recall**) of **92.86%**.

### 2. Unsupervised Patient Clustering

* Clustering was performed strictly on clinical features without exposing the ground-truth target[cite: 1].
* **K-Means Clustering ($k=2$):**
  * **Cluster 0 (Low-Risk Profile):** 183 patients; 22.40% disease prevalence.
  * **Cluster 1 (High-Risk Profile):** 121 patients; **80.99% disease prevalence** (captures **70.50%** of all true heart disease cases).
* **Agglomerative Hierarchical Clustering:**
  * **High-Risk Cluster:** 131 patients; **73.28% disease prevalence** (captures **69.06%** of all true heart disease cases).

---

## 🛠️ Repository Structure

```text
├── artifacts/                         # Serialized models, scalers, and evaluation plots
│   ├── confusion_matrices.png         # 2x2 confusion matrix grid
│   ├── roc_curves.png                 # Multi-model ROC comparison curves
│   ├── clustering_pca_plots.png       # 2D PCA cluster projections
│   ├── hierarchical_dendrogram.png    # Hierarchical tree dendrogram
│   └── *.joblib                       # Trained models, transformers, and metric dataframes
├── heart.csv                          # Cleveland Clinic dataset
├── phase1_preprocess.py               # Preprocessing, imputation, and feature scaling pipeline
├── phase2_supervised.py               # Supervised model training and benchmark evaluations
├── phase3_unsupervised.py             # K-Means, hierarchical clustering, and PCA reduction
├── app.py                             # Interactive Streamlit multi-tab clinical dashboard
├── requirements.txt                   # Project package dependencies
└── README.md                          # Project documentation
