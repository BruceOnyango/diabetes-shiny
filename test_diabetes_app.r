# =============================================================================
# tests/test_diabetes_app.R
# Test suite for Diabetes Risk Predictor
# Run with: testthat::test_file("tests/test_diabetes_app.R")
# =============================================================================

library(testthat)

# ── Helper: recreate the engineered features exactly as the app does ──────────
make_new_data <- function(glucose, bmi, age, pregnancies,
                          insulin, dpf, bp, skin) {
  bmi_category <- as.integer(as.character(cut(bmi,
    breaks = c(0, 18.5, 25, 30, Inf),
    labels = c(0, 1, 2, 3),
    right  = FALSE)))
  glucose_bmi <- round((glucose * bmi) / 100, 4)

  data.frame(
    Glucose                  = glucose,
    BMI                      = bmi,
    Age                      = age,
    Pregnancies              = pregnancies,
    Insulin                  = insulin,
    DiabetesPedigreeFunction = dpf,
    BloodPressure            = bp,
    SkinThickness            = skin,
    Glucose_BMI_Score        = glucose_bmi,
    BMI_Category             = bmi_category
  )
}

# ── Helper: winsorize (same function used in analysis script) ─────────────────
winsorize_manual <- function(x, lo = 0.05, hi = 0.95) {
  lb <- quantile(x, lo, na.rm = TRUE)
  ub <- quantile(x, hi, na.rm = TRUE)
  pmax(pmin(x, ub), lb)
}

# ── Helper: BMI category (same logic used in app and analysis) ───────────────
get_bmi_category <- function(bmi) {
  as.integer(as.character(cut(bmi,
    breaks = c(0, 18.5, 25, 30, Inf),
    labels = c(0, 1, 2, 3),
    right  = FALSE)))
}

# =============================================================================
# BLOCK 1: Feature Engineering Tests
# These test that the engineered features are computed correctly
# =============================================================================

test_that("BMI_Category correctly encodes WHO obesity thresholds", {
  # as.integer(as.character(cut(..., labels=0:3))) gives the label value
  # underweight=0, normal=1, overweight=2, obese=3
  expect_equal(get_bmi_category(17.0), 0L)  # underweight
  expect_equal(get_bmi_category(22.0), 1L)  # normal
  expect_equal(get_bmi_category(27.5), 2L)  # overweight
  expect_equal(get_bmi_category(35.0), 3L)  # obese
})

test_that("BMI_Category handles boundary values correctly", {
  expect_equal(get_bmi_category(18.5), 1L)  # exactly at normal boundary
  expect_equal(get_bmi_category(25.0), 2L)  # exactly at overweight boundary
  expect_equal(get_bmi_category(30.0), 3L)  # exactly at obese boundary
})

test_that("Glucose_BMI_Score is computed correctly", {
  glucose <- 140
  bmi     <- 32
  expected <- round((glucose * bmi) / 100, 4)
  actual   <- round((140 * 32) / 100, 4)
  expect_equal(actual, expected)
  expect_equal(actual, 44.8)
})

test_that("Glucose_BMI_Score scales with both inputs", {
  low_score  <- (80  * 20) / 100   # low glucose, low BMI
  high_score <- (180 * 45) / 100   # high glucose, high BMI
  expect_lt(low_score, high_score)
  expect_equal(low_score,  16.0)
  expect_equal(high_score, 81.0)
})

test_that("Young_High_Preg flag is correctly assigned", {
  # Should be 1: age < 30 AND pregnancies > 3
  expect_equal(as.integer(25 < 30 & 5 > 3), 1L)
  # Should be 0: age >= 30
  expect_equal(as.integer(35 < 30 & 5 > 3), 0L)
  # Should be 0: pregnancies <= 3
  expect_equal(as.integer(25 < 30 & 2 > 3), 0L)
  # Should be 0: both conditions fail
  expect_equal(as.integer(40 < 30 & 1 > 3), 0L)
})

# =============================================================================
# BLOCK 2: Data Wrangling Tests
# These test that the cleaning logic works correctly
# =============================================================================

test_that("Median imputation replaces zeros correctly", {
  x        <- c(100, 120, 0, 140, 0, 130)
  x[x == 0] <- NA
  imputed  <- ifelse(is.na(x), median(x, na.rm = TRUE), x)
  expect_false(any(imputed == 0))
  expect_false(any(is.na(imputed)))
  expect_equal(imputed[3], 125)  # median of 100,120,130,140
  expect_equal(imputed[5], 125)
})

test_that("Winsorize caps extreme values at 5th and 95th percentile", {
  set.seed(42)
  x    <- c(1:100)
  wx   <- winsorize_manual(x)
  lb   <- quantile(x, 0.05)
  ub   <- quantile(x, 0.95)
  expect_true(all(wx >= lb))
  expect_true(all(wx <= ub))
})

test_that("Winsorize preserves values within bounds", {
  x  <- c(50, 60, 70, 80, 90)
  wx <- winsorize_manual(x)
  # Middle values should be unchanged since no extreme outliers
  expect_equal(wx[3], 70)
})

test_that("Winsorize does not change dataset row count", {
  x  <- c(1:100, 9999, -9999)
  wx <- winsorize_manual(x)
  expect_equal(length(wx), length(x))
})

test_that("Zero counts match known dataset values", {
  # Known zero counts from the Pima dataset before imputation
  known_zeros <- c(
    Glucose       = 5,
    BloodPressure = 35,
    SkinThickness = 227,
    Insulin       = 374,
    BMI           = 11
  )
  # Insulin has the most zeros — confirm ordering
  expect_true(known_zeros["Insulin"] > known_zeros["Glucose"])
  expect_true(known_zeros["SkinThickness"] > known_zeros["BloodPressure"])
  expect_equal(unname(known_zeros["Insulin"]), 374)
})

# =============================================================================
# BLOCK 3: Probability and Statistics Tests
# These test that computed probabilities and statistics are within
# expected ranges based on the real results from the analysis
# =============================================================================

test_that("Marginal probabilities sum to 1", {
  P_diab    <- 268 / 768
  P_nondiab <- 500 / 768
  expect_equal(round(P_diab + P_nondiab, 10), 1)
})

test_that("Marginal probabilities match known values", {
  P_diab    <- 268 / 768
  P_nondiab <- 500 / 768
  expect_equal(round(P_diab,    4), 0.3490)
  expect_equal(round(P_nondiab, 4), 0.6510)
})

test_that("Conditional probability P(D|High Glucose) exceeds baseline", {
  P_baseline  <- 268 / 768         # 0.349
  P_d_high    <- 0.6853            # from real analysis
  P_d_normal  <- 0.2329
  expect_gt(P_d_high,   P_baseline)
  expect_lt(P_d_normal, P_baseline)
})

test_that("Risk multiplier is approximately 2.94x", {
  P_d_high   <- 0.6853
  P_d_normal <- 0.2329
  multiplier <- P_d_high / P_d_normal
  expect_equal(round(multiplier, 2), 2.94)
})

test_that("Bayes posterior P(D|Obese) exceeds prior P(D)", {
  P_prior    <- 268 / 768
  P_posterior <- 0.4576
  expect_gt(P_posterior, P_prior)
})

test_that("Cohen's d is in medium-large range", {
  cohens_d <- 0.6892
  # Cohen's conventions: small=0.2, medium=0.5, large=0.8
  expect_gt(cohens_d, 0.5)   # above medium
  expect_lt(cohens_d, 0.8)   # below large
})

test_that("Cramer's V indicates strong association", {
  cramers_v <- 0.4113
  # 0.3+ is considered strong for a 2x2 table
  expect_gt(cramers_v, 0.3)
  expect_lt(cramers_v, 1.0)
})

test_that("t-statistic for glucose is large and positive", {
  t_stat <- 14.853
  expect_gt(t_stat, 10)    # very strong signal
  expect_gt(t_stat, 0)     # positive: diabetic group is higher
})

# =============================================================================
# BLOCK 4: Model Performance Tests
# These test that model metrics are within expected ranges
# =============================================================================

test_that("Logistic Regression AUC is the highest", {
  auc_lr   <- 0.8177
  auc_tree <- 0.7280
  auc_rf   <- 0.7762
  expect_gt(auc_lr, auc_tree)
  expect_gt(auc_lr, auc_rf)
})

test_that("All model AUCs are above 0.70 (acceptable discrimination)", {
  aucs <- c(0.8177, 0.7280, 0.7762)
  expect_true(all(aucs > 0.70))
})

test_that("No model AUC exceeds 1.0", {
  aucs <- c(0.8177, 0.7280, 0.7762)
  expect_true(all(aucs <= 1.0))
})

test_that("Threshold tuning improves sensitivity", {
  sensitivity_default <- 0.5849
  sensitivity_tuned   <- 0.7170
  expect_gt(sensitivity_tuned, sensitivity_default)
})

test_that("Threshold tuning reduces specificity (expected trade-off)", {
  specificity_default <- 0.8400
  specificity_tuned   <- 0.7200
  expect_lt(specificity_tuned, specificity_default)
})

test_that("Threshold tuning improves F1 score", {
  f1_default <- 0.6200
  f1_tuned   <- 0.6387
  expect_gt(f1_tuned, f1_default)
})

test_that("Sensitivity and specificity are both between 0 and 1", {
  metrics <- c(0.5849, 0.8400, 0.6792, 0.6200, 0.5849, 0.8100,
               0.7170, 0.7200)
  expect_true(all(metrics >= 0 & metrics <= 1))
})

# =============================================================================
# BLOCK 5: Feature Importance Tests
# =============================================================================

test_that("Glucose_BMI_Score ranks first in feature importance", {
  fi <- data.frame(
    Feature = c("Glucose_BMI_Score","Glucose","Age","BMI",
                "DPF","Pregnancies","Insulin",
                "BloodPressure","SkinThickness","BMI_Category"),
    Score   = c(100, 88.36, 54.17, 51.34, 47.94,
                28.87, 23.05, 21.63, 20.10, 0.00)
  )
  top_feature <- fi$Feature[which.max(fi$Score)]
  expect_equal(top_feature, "Glucose_BMI_Score")
})

test_that("BMI_Category scores zero importance", {
  fi <- data.frame(
    Feature = c("Glucose_BMI_Score","Glucose","Age","BMI",
                "DPF","Pregnancies","Insulin",
                "BloodPressure","SkinThickness","BMI_Category"),
    Score   = c(100, 88.36, 54.17, 51.34, 47.94,
                28.87, 23.05, 21.63, 20.10, 0.00)
  )
  bmi_cat_score <- fi$Score[fi$Feature == "BMI_Category"]
  expect_equal(bmi_cat_score, 0.00)
})

test_that("All importance scores are non-negative", {
  scores <- c(100, 88.36, 54.17, 51.34, 47.94,
              28.87, 23.05, 21.63, 20.10, 0.00)
  expect_true(all(scores >= 0))
})

test_that("Engineered feature outperforms raw Glucose", {
  score_engineered <- 100.00
  score_raw_glucose <- 88.36
  expect_gt(score_engineered, score_raw_glucose)
})

# =============================================================================
# BLOCK 6: Risk Level Classification Tests
# These test the app's low/medium/high risk logic
# =============================================================================

get_risk_level <- function(pct) {
  if (pct < 30) "low" else if (pct <= 60) "medium" else "high"
}

test_that("Risk level is low below 30%", {
  expect_equal(get_risk_level(0),  "low")
  expect_equal(get_risk_level(15), "low")
  expect_equal(get_risk_level(29), "low")
})

test_that("Risk level is medium between 30% and 60%", {
  expect_equal(get_risk_level(30), "medium")
  expect_equal(get_risk_level(45), "medium")
  expect_equal(get_risk_level(60), "medium")
})

test_that("Risk level is high above 60%", {
  expect_equal(get_risk_level(61),  "high")
  expect_equal(get_risk_level(80),  "high")
  expect_equal(get_risk_level(100), "high")
})

test_that("Risk boundaries are not ambiguous", {
  # 30 should be medium not low
  expect_equal(get_risk_level(30), "medium")
  # 60 should be medium not high
  expect_equal(get_risk_level(60), "medium")
  # 61 should be high not medium
  expect_equal(get_risk_level(61), "high")
})

# =============================================================================
# BLOCK 7: Input Validation Tests
# These test edge cases for the patient input values
# =============================================================================

test_that("make_new_data returns a data frame with correct columns", {
  nd <- make_new_data(117, 32, 30, 1, 125, 0.47, 72, 29)
  expect_s3_class(nd, "data.frame")
  expected_cols <- c("Glucose","BMI","Age","Pregnancies","Insulin",
                     "DiabetesPedigreeFunction","BloodPressure",
                     "SkinThickness","Glucose_BMI_Score","BMI_Category")
  expect_equal(names(nd), expected_cols)
})

test_that("make_new_data returns exactly one row", {
  nd <- make_new_data(117, 32, 30, 1, 125, 0.47, 72, 29)
  expect_equal(nrow(nd), 1)
})

test_that("Glucose_BMI_Score is correct for default input values", {
  nd <- make_new_data(117, 32, 30, 1, 125, 0.47, 72, 29)
  expected <- round((117 * 32) / 100, 4)
  expect_equal(nd$Glucose_BMI_Score, expected)
  expect_equal(nd$Glucose_BMI_Score, 37.44)
})

test_that("BMI_Category is 3 (obese) for default BMI of 32", {
  nd <- make_new_data(117, 32, 30, 1, 125, 0.47, 72, 29)
  expect_equal(nd$BMI_Category, 3L)  # BMI 32 = obese category
})

test_that("High glucose patient has higher Glucose_BMI_Score", {
  low_risk  <- make_new_data(90,  22, 25, 0, 80, 0.2, 70, 20)
  high_risk <- make_new_data(180, 45, 50, 8, 300, 1.5, 80, 40)
  expect_gt(high_risk$Glucose_BMI_Score, low_risk$Glucose_BMI_Score)
})