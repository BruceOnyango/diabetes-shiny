# Diabetes Risk Prediction

---

## Live Demo

<a href="https://bruceonyango.link/diabetes" target="_blank">bruceonyango.link/diabetes</a>

---

## Overview

A complete end-to-end data science pipeline predicting diabetes risk in Pima Indian women aged 21 and above, using eight routine clinical measurements. The final product is a deployed Shiny web application with five tabs serving both clinical and executive audiences.

**Problem type:** Supervised binary classification (Diabetic / Non-Diabetic)  
**Dataset:** Pima Indians Diabetes Dataset — 768 observations, UCI Machine Learning Repository  
**Language:** R 4.3.3  
**Best model:** Logistic Regression — AUC 0.818

---

## Repository Structure

```
pima-diabetes-risk-prediction/
├── app.R                    # Shiny application (5 tabs)
├── diabetes_analysis.R      # Full analysis pipeline — run this first
├── diabetes_cleaned.csv     # Cleaned dataset with engineered features
├── logistic_model.rds       # Saved logistic regression model
├── roc_data.rds             # Saved ROC curve data for Shiny app
├── tests/
│   └── test_diabetes_app.R  # 71 unit tests (testthat)
└── README.md
```

---

## How to Run

### Step 1 — Install packages

```r
install.packages(c(
  "ggplot2", "dplyr", "reshape2", "corrplot",
  "caret", "randomForest", "rpart", "pROC",
  "e1071", "shiny", "scales", "testthat"
))
```

### Step 2 — Run the analysis script

Open `diabetes_analysis.R` in RStudio, set your working directory to the project folder, then press **Ctrl + A** followed by **Ctrl + Enter**.

This creates:
- `diabetes_cleaned.csv` — cleaned dataset with engineered features
- `logistic_model.rds` — trained model for the Shiny app
- `roc_data.rds` — saved test predictions for real ROC curves
- `real_results.txt` — all computed statistics
- 8 plot PNG files

### Step 3 — Launch the Shiny app

Open `app.R` and click the **Run App** button in RStudio, or run:

```r
shiny::runApp()
```

### Step 4 — Run tests

```r
library(testthat)
test_file("tests/test_diabetes_app.R")
```

Expected output:
```
v |         71 | diabetes_app
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 71 ]
```

---

## Dataset

| Variable | Type | Description |
|---|---|---|
| Pregnancies | Integer | Number of times pregnant |
| Glucose | Integer | 2-hour plasma glucose (mg/dL) |
| BloodPressure | Integer | Diastolic blood pressure (mm Hg) |
| SkinThickness | Integer | Triceps skinfold thickness (mm) |
| Insulin | Integer | 2-hour serum insulin (mu U/ml) |
| BMI | Numeric | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Numeric | Genetic diabetes risk score |
| Age | Integer | Age in years |
| Outcome | Factor | Diabetic / Non.Diabetic |
| BMI_Category | Integer | WHO obesity threshold (0-3) — engineered |
| Glucose_BMI_Score | Numeric | (Glucose × BMI) / 100 — engineered |
| Young_High_Preg | Integer | Age < 30 AND Pregnancies > 3 — engineered |

**Class distribution:** 500 Non-Diabetic (65.1%) / 268 Diabetic (34.9%)

---

## Data Wrangling

### Missing Values

Five columns used zero as a biological impossibility placeholder:

| Column | Zeros | Imputed With |
|---|---|---|
| Glucose | 5 | Median = 117 mg/dL |
| BloodPressure | 35 | Median = 72 mm Hg |
| SkinThickness | 227 | Median = 29 mm |
| Insulin | 374 | Median = 125 mu U/ml |
| BMI | 11 | Median = 32.3 kg/m² |

Median chosen over mean because Insulin is heavily right-skewed (48.7% missing).

### Outlier Handling

Manual IQR Winsorizing at the 5th and 95th percentile applied to `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `BloodPressure`. All 768 rows preserved.

> **Note:** `DescTools::Winsorize()` throws `unused argument (probs = ...)` in some R versions. Use the manual implementation in the script instead.

```r
winsorize_manual <- function(x, lo = 0.05, hi = 0.95) {
  pmax(pmin(x, quantile(x, hi)), quantile(x, lo))
}
```

### Feature Engineering

| Feature | Formula | Clinical Rationale |
|---|---|---|
| `BMI_Category` | WHO thresholds 0-3 | Captures threshold-based obesity risk |
| `Glucose_BMI_Score` | `(Glucose × BMI) / 100` | Compound metabolic risk — strongest predictor |
| `Young_High_Preg` | `Age < 30 AND Pregnancies > 3` | Elevated gestational diabetes risk |

---

## Key Results

### Summary Statistics

| Metric | Non-Diabetic | Diabetic | Difference |
|---|---|---|---|
| Mean Glucose (mg/dL) | 110.68 | 142.13 | +31.45 |
| Mean BMI | 30.93 | 34.94 | +4.01 |
| Mean Age (years) | 31.19 | 37.07 | +5.88 |
| Mean Insulin (mu U/ml) | 124.15 | 152.28 | +28.13 |
| Mean Blood Pressure | 70.85 | 74.60 | +3.75 |

### Correlations with Outcome

| Feature | r |
|---|---|
| Glucose_BMI_Score | **0.5246** |
| Glucose | 0.4928 |
| BMI_Category | 0.3105 |
| BMI | 0.3085 |
| Insulin | 0.2387 |
| Age | 0.2384 |

### Hypothesis Tests

| Test | Statistic | p-value | Effect Size | Decision |
|---|---|---|---|---|
| Glucose t-test (one-tailed) | t = 14.853 | 1.77 × 10⁻⁴¹ | — | Reject H₀ |
| BMI t-test (two-tailed) | t = 9.238 | 4.44 × 10⁻¹⁹ | Cohen's d = 0.689 | Reject H₀ |
| Chi-square: Glucose vs Outcome | χ² = 129.94 | 4.23 × 10⁻³⁰ | Cramer's V = 0.411 | Reject H₀ |

### Probability Analysis

| Measure | Value |
|---|---|
| P(Diabetic) baseline | 0.3490 |
| P(Diabetic \| Glucose ≥ 140) | **0.6853** |
| P(Diabetic \| Glucose < 140) | 0.2329 |
| Risk multiplier | **2.94×** |
| P(Diabetic \| Obese) via Bayes | 0.4576 |

A glucose reading at or above 140 mg/dL shifts the diabetes probability from 34.9% to 68.5% — a 2.94× risk multiplier. This is the basis for the primary screening recommendation.

### Model Comparison

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|
| AUC | **0.8177** | 0.7280 | 0.7762 |
| Accuracy | 75.2% | 64.1% | 73.2% |
| Sensitivity | 58.5% | 67.9% | 58.5% |
| Specificity | **84.0%** | 62.0% | 81.0% |
| F1-Score | **0.620** | 0.567 | 0.602 |

**Logistic Regression selected** — highest AUC and best sensitivity-specificity balance.

### Threshold Tuning

| Metric | Default (0.50) | Tuned (0.40) |
|---|---|---|
| Sensitivity | 58.5% | **71.7%** |
| Specificity | 84.0% | 72.0% |
| F1-Score | 0.620 | **0.639** |

Lowering the threshold to 0.40 catches 7 in every 10 true diabetic patients — justified by the asymmetric cost of a missed diagnosis versus a false alarm.

### Feature Importance (Random Forest)

| Rank | Feature | Score |
|---|---|---|
| 1 | Glucose_BMI_Score | 100.00 |
| 2 | Glucose | 88.36 |
| 3 | Age | 54.17 |
| 4 | BMI | 51.34 |
| 5 | DiabetesPedigreeFunction | 47.94 |
| 6 | Pregnancies | 28.87 |
| 7 | Insulin | 23.05 |
| 8 | BloodPressure | 21.63 |
| 9 | SkinThickness | 20.10 |
| 10 | BMI_Category | 0.00 |

---

## Shiny App

The app has five tabs:

| Tab | Audience | Content |
|---|---|---|
| Executive Summary | C-suite, policymakers | KPI cards, plain-language findings, screening efficiency chart |
| Risk Predictor | Clinicians, nurses | Live prediction tool using the trained model |
| Data Overview | Analysts | Class distribution, mean comparisons, interactive histogram |
| Risk Factors | Analysts | Interactive scatter plot, feature importance, age density |
| Model Performance | Data scientists | Real ROC curves, threshold tuning, model comparison |

---

## Tests

71 unit tests across 7 blocks using `testthat`:

| Block | Tests | Covers |
|---|---|---|
| Feature Engineering | 12 | BMI_Category boundaries, Glucose_BMI_Score formula, Young_High_Preg logic |
| Data Wrangling | 10 | Median imputation, Winsorize bounds, row preservation |
| Probability & Statistics | 9 | Marginal probs, conditional prob shift, Bayes, Cohen's d, Cramer's V |
| Model Performance | 7 | AUC ranking, threshold tuning sensitivity/F1/specificity trade-off |
| Feature Importance | 4 | Engineered feature ranks first, BMI_Category is zero |
| Risk Classification | 4 | Low/medium/high boundaries, no ambiguity at cut-points |
| Input Validation | 5 | Data frame structure, correct columns, edge cases |

---

## Recommendations

| Finding | Decision | Rationale |
|---|---|---|
| P(D\|Glucose ≥ 140) = 0.685 | Use 140 mg/dL as primary screening threshold | 2.94× risk multiplier |
| Compound risk P_joint = 0.152 | Joint metabolic clinic referral | Glucose_BMI_Score ranks #1 |
| Age risk accelerates at 35 | Age-stratified screening frequency | Non-linear age density |
| LR AUC = 0.818 | Deploy as triage tool at threshold 0.40 | Highest AUC; interpretable |
| Glucose_BMI_Score ranks #1 | Use as standalone screening indicator | Computable from two measurements |
| 65/35 class imbalance | Evaluate on Sensitivity and F1, not accuracy | Naive classifier scores 65.1% |

---

## Limitations

- Dataset is specific to **Pima Indian women aged 21+** — do not generalise to other populations without retraining
- **48.7% of Insulin values** were imputed with the same median constant, contributing no individual-patient signal for nearly half the sample
- **BMI_Category scored zero** in Random Forest importance — redundant when continuous BMI is present
- All statistics are from a single 80/20 stratified split with seed 42

---

*All statistics were computed by running `diabetes_analysis.R` against the 768-record dataset. No values were estimated or fabricated.*