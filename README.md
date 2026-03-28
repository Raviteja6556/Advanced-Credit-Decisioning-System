# 💳 Advanced Credit Decisioning System
### Production-Grade Loan Default Prediction — LightGBM · Optuna · SHAP · Fairlearn · CalibratedClassifierCV

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-green?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-3.6+-purple?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-0.45+-red?style=flat-square)
![Fairlearn](https://img.shields.io/badge/Fairlearn-0.10+-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-blue?style=flat-square&logo=scikitlearn)

---

## 🎯 What This System Does

Credit decisioning is one of the highest-stakes ML applications — a wrong approval costs money, a wrong rejection loses a customer. This system goes far beyond a standard binary classifier.

It is a **deployment-ready credit risk engine** that:
- Predicts the **probability of default (PD)** for any loan applicant
- Makes **Approve / Reject decisions** at a business-profit-optimized threshold
- Explains **why** every decision was made — globally and locally via SHAP
- Audits decisions for **geographic fairness** via Fairlearn
- Persists all artifacts for **zero-retraining inference** in production

---

## 🏗️ System Architecture & Pipeline

```
Lending Club Dataset (887K+ loans)
          │
          ▼
┌─────────────────────────┐
│   Data Ingestion &       │
│   Leakage Elimination    │  ← 36 post-origination columns dropped
│   Missing Value Handling │  ← column & row-level thresholding
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Target Engineering     │  ← is_bad_loan (Charged Off / Default)
│   Stratified Sampling    │  ← 200K stratified sample
│   Class Rebalancing      │  ← 65% good / 35% bad (under + oversampling)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Train / Val / Test     │  ← 75% / 15% / 10% stratified splits
│   EDA (Uni + Bivariate)  │  ← t-tests, Chi-square, Cramér's V
│   Multicollinearity Check│  ← VIF → dropped policy_code, funded_amnt_inv,
│                          │         installment, pub_rec
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Preprocessing Pipeline │
│   Numerical: Impute      │  ← SimpleImputer (median)
│             + Scale      │  ← RobustScaler
│   Categorical: Impute    │  ← SimpleImputer (most_frequent)
│              + Encode    │  ← OneHotEncoder
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Model Training         │
│   Baseline: LogReg       │  ← ROC-AUC 0.74 | Brier 0.19
│   Champion: LightGBM     │  ← Optuna HPO (50 trials, maximize AUC)
│             ROC-AUC 0.91 │  ← Brier Score 0.12
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Calibration            │  ← CalibratedClassifierCV (isotonic)
│   Threshold Optimization │  ← Business profit matrix
│   Optimal Threshold 0.19 │  ← $8.74M max expected profit
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Explainability (SHAP)  │
│   Global: beeswarm + bar │  ← Feature importance across dataset
│   Local: force plots     │  ← Per-applicant decision reasoning
│   All 4 quadrants        │  ← TP, TN, FP, FN explained
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Fairness Audit         │  ← Fairlearn MetricFrame
│   Sensitive: addr_state  │  ← Geographic cohort analysis
│   Disparate Impact: 0.41 │  ← Selection rate disparity
│   TPR Disparity: 0.78    │  ← Equal opportunity gap
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Artifact Persistence   │  ← joblib serialization
│   Inference System       │  ← Standalone deployment-ready function
└─────────────────────────┘
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| **Core ML** | LightGBM, Scikit-learn, LogisticRegression |
| **HPO** | Optuna (50 trials, TPE sampler, ROC-AUC objective) |
| **Calibration** | CalibratedClassifierCV — isotonic regression |
| **Explainability** | SHAP (TreeExplainer, force plots, beeswarm) |
| **Fairness** | Fairlearn (MetricFrame, selection_rate, TPR) |
| **Preprocessing** | RobustScaler, OneHotEncoder, SimpleImputer, ColumnTransformer |
| **Resampling** | sklearn.utils.resample (under + oversampling) |
| **Persistence** | joblib |
| **EDA** | Pandas, Seaborn, Matplotlib, Missingno, ydata-profiling |
| **Statistics** | SciPy (t-tests, Chi-square, Cramér's V, Shapiro-Wilk, Levene) |
| **Dataset** | Lending Club Loan Data (Kaggle) — 887K+ loans |

---

## 📊 Model Performance

| Model | ROC-AUC | Avg Precision | F1-Score | Brier Score |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.7440 | 0.5819 | 0.4804 | 0.1908 |
| LightGBM Champion (Val) | 0.9134 | 0.8455 | 0.7655 | 0.1206 |
| Calibrated LightGBM (Val) | 0.9141 | 0.8415 | 0.7558 | 0.1108 |
| **Calibrated LightGBM (Test)** | **0.9113** | **0.8349** | **0.7527** | **0.1129** |

**Optimal Decision Threshold:** 0.1902
**Maximum Expected Profit:** $6,743,000

---

## 🧩 Key Engineering Decisions

### 1. Data Leakage Elimination
36 post-origination columns were identified and surgically removed — fields like `total_pymnt`, `recoveries`, `last_pymnt_amnt` that would only be known after loan outcome. Without this step, any model would achieve near-perfect AUC but fail completely in production.

### 2. VIF-Based Multicollinearity Pruning
Variance Inflation Factor analysis identified and removed `policy_code`, `funded_amnt_inv`, `installment`, and `pub_rec` — collinear features that inflate model coefficients without adding predictive signal.

### 3. Optuna Hyperparameter Optimization
50 Optuna trials on the validation set using Tree-structured Parzen Estimator (TPE) sampler, maximizing ROC-AUC. LightGBM's native early stopping was used within each trial to prevent overfitting — resulting in a 23% relative AUC improvement over the Logistic Regression baseline.

### 4. Isotonic Probability Calibration
Raw LightGBM outputs are not inherently well-calibrated probabilities. `CalibratedClassifierCV` with isotonic regression was applied on the validation set — producing reliable probability of default (PD) estimates essential for business threshold optimization.

### 5. Business-Profit-Matrix Threshold Optimization
Rather than defaulting to 0.5, the decision threshold was selected by maximizing expected profit using a business cost matrix:
- **True Negative (correct approval):** revenue gained
- **False Positive (wrong approval):** loan loss
- **True Positive (correct rejection):** avoided loss
- **False Negative (wrong rejection):** opportunity cost

Result: threshold 0.19 → **$8.74M maximum expected profit** on test set.

### 6. Four-Quadrant SHAP Explainability
Local SHAP force plots were generated for representative samples from all four prediction quadrants (TP, TN, FP, FN) — not just correct predictions. This gives the system auditability for wrong decisions, critical for regulatory compliance in credit.

### 7. Geographic Fairness Audit
`Fairlearn MetricFrame` was used to assess disparate impact and equal opportunity across U.S. states (`addr_state`). Findings:
- **Disparate Impact Ratio: 0.41** — approval rates vary significantly across states
- **TPR Disparity: 0.78** — true positive rates not equal across geographic cohorts
These findings surface actionable bias mitigation pathways for responsible deployment.

---

## 🗂️ Persisted Artifacts

All artifacts are saved via `joblib` to `model_artifacts/`:

```
model_artifacts/
├── preprocessing_pipeline.joblib       ← Fitted ColumnTransformer
├── calibrated_lightgbm_model.joblib    ← CalibratedClassifierCV champion
├── shap_explainer.joblib               ← Fitted TreeExplainer
├── optimal_decision_threshold.joblib   ← 0.1902
├── X_train_columns_for_inference.joblib← Feature column order
├── numerical_features.joblib           ← Numerical feature list
└── categorical_features.joblib         ← Categorical feature list
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
Kaggle API credentials (for dataset download)
```

### Installation
```bash
git clone https://github.com/Raviteja6556/advanced-credit-decisioning-system.git
cd advanced-credit-decisioning-system
pip install -r requirements.txt
```

### Run Inference on New Applicant
```python
import joblib
import pandas as pd

# Load artifacts
preprocessor = joblib.load('model_artifacts/preprocessing_pipeline.joblib')
model = joblib.load('model_artifacts/calibrated_lightgbm_model.joblib')
explainer = joblib.load('model_artifacts/shap_explainer.joblib')
threshold = joblib.load('model_artifacts/optimal_decision_threshold.joblib')

# New applicant data
applicant = {
    'loan_amnt': 10000.0,
    'term': ' 36 months',
    'int_rate': 7.5,
    'grade': 'A',
    'annual_inc': 75000.0,
    'dti': 15.0,
    # ... remaining features
}

result = inference_system(applicant, preprocessor, model, explainer, threshold)
print(result['loan_decision'])           # 'Approve' or 'Reject'
print(result['probability_of_default']) # e.g. 0.0823
print(result['top_shap_reasons'])       # Top 5 features driving decision
```

---

## 💬 Inference System Output

```json
{
  "loan_decision": "Approve",
  "probability_of_default": 0.0823,
  "top_shap_reasons": [
    {"feature": "int_rate", "shap_value": -0.312},
    {"feature": "grade_A", "shap_value": -0.289},
    {"feature": "dti", "shap_value": -0.156},
    {"feature": "revol_util", "shap_value": -0.134},
    {"feature": "annual_inc", "shap_value": -0.098}
  ]
}
