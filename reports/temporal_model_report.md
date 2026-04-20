# Temporal Credit Risk Modeling Report

## Objective And Leakage Rule

This report documents a chronologically valid probability-of-default workflow.
The project uses issue_date only as temporal metadata and never as a predictive feature.
Grid search and model selection are restricted to the training and validation periods.
The final holdout test period is opened only after the model family, features, and hyperparameters are locked.

## Dataset And Temporal Split

- Processed dataset path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/loan_clean.csv`
- Prediction export path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/test_with_pd_best_model.csv`
- Report sample fraction: `1.0`
- Grid search enabled: `True`

| partition | rows | default_rate | min_issue_date | max_issue_date |
| --- | --- | --- | --- | --- |
| train | 826604 | 0.1843 | 2007-06-01 | 2015-12-01 |
| validation | 293095 | 0.2328 | 2016-01-01 | 2016-12-01 |
| test | 225611 | 0.2128 | 2017-01-01 | 2018-12-01 |

## Cleaning Method

- Filter the target to Fully Paid and Charged Off loans only.
- Parse issue_d into issue_date and keep it strictly as temporal metadata.
- Parse emp_length into numeric years and keep a missing-value flag.
- Extract term_months from the original term string.
- Convert interest-rate and revolving-utilization percent strings into numeric values.
- Convert invalid DTI values below 0 or above 80 to missing before imputation.
- Convert invalid int_rate below 0 or above 40 to missing before imputation.
- Convert invalid revol_util below 0 or above 150 to missing before imputation.
- Impute all modeled numeric columns with the median inside the processed dataset.
- One-hot encode home_ownership, purpose, fico_bucket, verification_status, grade, initial_list_status, and application_type with drop_first=True.

## Selected Features

- Core numeric features: `loan_amnt, annual_inc, fico_range_low, dti, emp_length, term_months`
- Additional numeric features: `int_rate, installment, delinq_2yrs, inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util, total_acc, mort_acc, pub_rec_bankruptcies`
- Base engineered features: `missing_emp_length_flag, log_annual_inc, loan_to_income, high_dti_flag, long_term_flag`
- Extended engineered features: `installment_to_income, revol_bal_to_income, high_revol_util_flag, recent_inquiry_flag, prior_delinquency_flag, bankruptcy_flag`
- Dummy groups available: `home_ownership_*, purpose_*, fico_bucket_*, verification_status_*, grade_*, initial_list_status_*, application_type_*`
- Active feature flags: `use_additional_numeric_features, use_engineered_features, use_extended_engineered_features, use_fico_bucket_dummies, use_home_ownership_dummies, use_purpose_dummies, use_verification_status_dummies`
- Final modeled column count: `51`

### Encoded Feature Columns

loan_amnt, annual_inc, fico_range_low, dti, emp_length, term_months, int_rate, installment, delinq_2yrs, inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util, total_acc, mort_acc, pub_rec_bankruptcies, missing_emp_length_flag, log_annual_inc, loan_to_income, high_dti_flag, long_term_flag, installment_to_income, revol_bal_to_income, high_revol_util_flag, recent_inquiry_flag, prior_delinquency_flag, bankruptcy_flag, home_ownership_MORTGAGE, home_ownership_NONE, home_ownership_OTHER, home_ownership_OWN, home_ownership_RENT, purpose_credit_card, purpose_debt_consolidation, purpose_educational, purpose_home_improvement, purpose_house, purpose_major_purchase, purpose_medical, purpose_moving, purpose_other, purpose_renewable_energy, purpose_small_business, purpose_vacation, purpose_wedding, fico_bucket_good, fico_bucket_subprime, fico_bucket_very_good, verification_status_Source Verified, verification_status_Verified

## Candidate Models And Search Space

- Enabled model families: `logistic_regression, random_forest, hist_gradient_boosting, xgboost`
- Grid search models: `logistic_regression, random_forest, hist_gradient_boosting, xgboost`

### logistic_regression

Base parameters:

```json
{
  "C": 1.0,
  "max_iter": 1000,
  "class_weight": null
}
```

Grid search space:

```json
{
  "C": [
    0.5,
    2.0
  ],
  "class_weight": [
    null,
    "balanced"
  ]
}
```

### random_forest

Base parameters:

```json
{
  "n_estimators": 120,
  "max_depth": 14,
  "min_samples_split": 100,
  "min_samples_leaf": 25,
  "max_features": "sqrt",
  "n_jobs": -1,
  "random_state": 42
}
```

Grid search space:

```json
{
  "n_estimators": [
    80,
    120
  ],
  "max_depth": [
    10,
    14
  ]
}
```

### hist_gradient_boosting

Base parameters:

```json
{
  "learning_rate": 0.05,
  "max_iter": 150,
  "max_depth": 6,
  "max_leaf_nodes": 31,
  "min_samples_leaf": 50,
  "l2_regularization": 0.0,
  "early_stopping": false
}
```

Grid search space:

```json
{
  "learning_rate": [
    0.03,
    0.05
  ],
  "max_depth": [
    4,
    6
  ]
}
```

### xgboost

Base parameters:

```json
{
  "n_estimators": 120,
  "max_depth": 6,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "min_child_weight": 5,
  "reg_lambda": 1.0,
  "random_state": 42,
  "n_jobs": -1
}
```

Grid search space:

```json
{
  "n_estimators": [
    80,
    120
  ],
  "max_depth": [
    4,
    6
  ]
}
```

## Validation Ranking

| validation_rank | model_name | search_mode | search_candidates | params | validation_auc | validation_ks | validation_brier | calibration_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost | grid_search | 4 | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7151 | 0.3123 | 0.1628 | Visible calibration drift; inspect the curve before finalizing. |
| 2 | HistGradientBoosting | grid_search | 4 | {"early_stopping": false, "l2_regularization": 0.0, "learning_rate": 0.05, "max_depth": 6, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 50} | 0.7151 | 0.3123 | 0.1628 | Underestimates risk in the highest-PD bins. |
| 3 | Logistic Regression | grid_search | 4 | {"C": 2.0, "class_weight": "balanced", "max_iter": 1000} | 0.7115 | 0.3070 | 0.2118 | Overestimates risk in the highest-PD bins. |
| 4 | Random Forest | grid_search | 4 | {"max_depth": 14, "max_features": "sqrt", "min_samples_leaf": 25, "min_samples_split": 100, "n_estimators": 120, "n_jobs": -1, "random_state": 42} | 0.7111 | 0.3056 | 0.1641 | Underestimates risk in the highest-PD bins. |

## Grid Search Results

| model_name | search_rank | params | validation_auc | validation_ks | validation_brier |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1 | {"C": 2.0, "class_weight": "balanced", "max_iter": 1000} | 0.7115 | 0.3070 | 0.2118 |
| Logistic Regression | 2 | {"C": 0.5, "class_weight": "balanced", "max_iter": 1000} | 0.7115 | 0.3070 | 0.2118 |
| Logistic Regression | 3 | {"C": 2.0, "class_weight": null, "max_iter": 1000} | 0.7112 | 0.3067 | 0.1636 |
| Logistic Regression | 4 | {"C": 0.5, "class_weight": null, "max_iter": 1000} | 0.7112 | 0.3067 | 0.1636 |
| Random Forest | 1 | {"max_depth": 14, "max_features": "sqrt", "min_samples_leaf": 25, "min_samples_split": 100, "n_estimators": 120, "n_jobs": -1, "random_state": 42} | 0.7111 | 0.3056 | 0.1641 |
| Random Forest | 2 | {"max_depth": 14, "max_features": "sqrt", "min_samples_leaf": 25, "min_samples_split": 100, "n_estimators": 80, "n_jobs": -1, "random_state": 42} | 0.7109 | 0.3057 | 0.1641 |
| Random Forest | 3 | {"max_depth": 10, "max_features": "sqrt", "min_samples_leaf": 25, "min_samples_split": 100, "n_estimators": 120, "n_jobs": -1, "random_state": 42} | 0.7079 | 0.3010 | 0.1653 |
| Random Forest | 4 | {"max_depth": 10, "max_features": "sqrt", "min_samples_leaf": 25, "min_samples_split": 100, "n_estimators": 80, "n_jobs": -1, "random_state": 42} | 0.7076 | 0.3005 | 0.1653 |
| HistGradientBoosting | 1 | {"early_stopping": false, "l2_regularization": 0.0, "learning_rate": 0.05, "max_depth": 6, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 50} | 0.7151 | 0.3123 | 0.1628 |
| HistGradientBoosting | 2 | {"early_stopping": false, "l2_regularization": 0.0, "learning_rate": 0.05, "max_depth": 4, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 50} | 0.7137 | 0.3101 | 0.1633 |
| HistGradientBoosting | 3 | {"early_stopping": false, "l2_regularization": 0.0, "learning_rate": 0.03, "max_depth": 6, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 50} | 0.7130 | 0.3098 | 0.1634 |
| HistGradientBoosting | 4 | {"early_stopping": false, "l2_regularization": 0.0, "learning_rate": 0.03, "max_depth": 4, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 50} | 0.7112 | 0.3068 | 0.1641 |
| XGBoost | 1 | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7151 | 0.3123 | 0.1628 |
| XGBoost | 2 | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 80, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7132 | 0.3097 | 0.1634 |
| XGBoost | 3 | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7129 | 0.3093 | 0.1636 |
| XGBoost | 4 | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "n_estimators": 80, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7105 | 0.3051 | 0.1644 |

## Feature Discovery On Omitted Feature Groups

- Feature-search model: `XGBoost`
- Baseline validation AUC: `0.7151`
- Baseline validation KS: `0.3123`
- Baseline validation Brier: `0.1628`

| feature_rank | feature_group | feature_flag | search_model | search_mode | params | added_feature_count | added_columns_preview | validation_auc | validation_ks | validation_brier | delta_auc | delta_ks | delta_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Application type dummies | use_application_type_dummies | XGBoost | grid_search | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 1 | application_type_Joint App | 0.7152 | 0.3122 | 0.1628 | 0.0001 | -0.0001 | -0.0000 |
| 2 | Initial list status dummies | use_initial_list_status_dummies | XGBoost | grid_search | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 1 | initial_list_status_w | 0.7150 | 0.3132 | 0.1626 | -0.0002 | 0.0009 | -0.0002 |
| 3 | Grade dummies | use_grade_dummies | XGBoost | grid_search | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 6 | grade_B, grade_C, grade_D, grade_E, grade_F, grade_G | 0.7148 | 0.3112 | 0.1629 | -0.0003 | -0.0011 | 0.0001 |

## Final Chosen Hyperparameters

- Selected model: `XGBoost`
- Validation AUC: `0.7151`
- Validation KS: `0.3123`
- Validation Brier: `0.1628`

```json
{"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8}
```

## Holdout Test Results

- Champion test AUC: `0.7106`
- Champion test KS: `0.3063`
- Champion test Brier: `0.1521`

| validation_rank | model_name | params | test_auc | test_ks | test_brier | calibration_summary |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7106 | 0.3063 | 0.1521 | Well aligned across calibration bins. |
| 2 | HistGradientBoosting | {"early_stopping": false, "l2_regularization": 0.0, "learning_rate": 0.05, "max_depth": 6, "max_iter": 150, "max_leaf_nodes": 31, "min_samples_leaf": 50} | 0.7112 | 0.3065 | 0.1520 | Underestimates risk in the highest-PD bins. |
| 3 | Logistic Regression | {"C": 2.0, "class_weight": "balanced", "max_iter": 1000} | 0.7035 | 0.2975 | 0.2236 | Overestimates risk in the highest-PD bins. |
| 4 | Random Forest | {"max_depth": 14, "max_features": "sqrt", "min_samples_leaf": 25, "min_samples_split": 100, "n_estimators": 120, "n_jobs": -1, "random_state": 42} | 0.7073 | 0.3010 | 0.1527 | Underestimates risk in the highest-PD bins. |

## Champion Temporal Performance

| period | fitted_on | rows | default_rate | min_issue_date | max_issue_date | auc | ks | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | train only | 826604 | 0.1843 | 2007-06-01 | 2015-12-01 | 0.7258 | 0.3273 | 0.1346 |
| validation | train only | 293095 | 0.2328 | 2016-01-01 | 2016-12-01 | 0.7151 | 0.3123 | 0.1628 |
| test | train + validation | 225611 | 0.2128 | 2017-01-01 | 2018-12-01 | 0.7106 | 0.3063 | 0.1521 |

## Feature Importance And Interpretability

### Logistic Regression Coefficients

| feature | coefficient | abs_coefficient |
| --- | --- | --- |
| installment_to_income | -0.4204 | 0.4204 |
| loan_to_income | 0.3805 | 0.3805 |
| int_rate | 0.3622 | 0.3622 |
| home_ownership_MORTGAGE | -0.2510 | 0.2510 |
| fico_range_low | -0.2314 | 0.2314 |
| installment | 0.1751 | 0.1751 |
| dti | 0.1701 | 0.1701 |
| long_term_flag | 0.1633 | 0.1633 |
| term_months | 0.1633 | 0.1633 |
| home_ownership_RENT | -0.1350 | 0.1350 |
| log_annual_inc | -0.1327 | 0.1327 |
| home_ownership_OWN | -0.1185 | 0.1185 |
| open_acc | 0.1146 | 0.1146 |
| missing_emp_length_flag | 0.1054 | 0.1054 |
| mort_acc | -0.0939 | 0.0939 |

### XGBoost Feature Importance

| feature | importance |
| --- | --- |
| int_rate | 0.3292 |
| long_term_flag | 0.1601 |
| term_months | 0.0946 |
| home_ownership_MORTGAGE | 0.0266 |
| mort_acc | 0.0265 |
| home_ownership_RENT | 0.0233 |
| fico_range_low | 0.0229 |
| dti | 0.0206 |
| missing_emp_length_flag | 0.0192 |
| installment_to_income | 0.0181 |
| loan_to_income | 0.0168 |
| high_dti_flag | 0.0167 |
| verification_status_Source Verified | 0.0158 |
| fico_bucket_very_good | 0.0141 |
| inq_last_6mths | 0.0129 |

## Candidate Feature Flags

- `use_additional_numeric_features`: Additional origination-time numeric credit features
- `use_extended_engineered_features`: Extended engineered ratios and delinquency flags
- `use_verification_status_dummies`: Verification status dummies
- `use_grade_dummies`: Grade dummies
- `use_initial_list_status_dummies`: Initial list status dummies
- `use_application_type_dummies`: Application type dummies
