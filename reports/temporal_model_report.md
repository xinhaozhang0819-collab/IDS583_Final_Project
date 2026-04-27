# Temporal Credit Risk Modeling Report

## Objective And Leakage Rule

This report documents a chronologically valid probability-of-default workflow.
The project uses issue_date only as temporal metadata and never as a predictive feature.
Grid search and model selection are restricted to the training and validation periods.
The final holdout test period is opened only after the model family, features, and hyperparameters are locked.

Dual-PD update: the final expected-loss workflow uses the rerun original static HistGradientBoosting loan-level model for lifetime PD and uses a calibrated direct active-snapshot XGBoost model for twelve-month PD. The calendar-time hazard report below is retained as a comparison group and research diagnostic, not as the main twelve-month EL input.

## Direct Active-Snapshot 12M Champion

- Direct 12M model key: `direct_12m_xgboost`
- Candidate search: 6 candidates (1 Logistic baseline, 1 Random Forest benchmark, 4 XGBoost candidates)
- Selected direct 12M model: `XGBoost`
- Selected parameters: `n_estimators=120`, `max_depth=5`, `learning_rate=0.06`
- Direct 12M validation AUC: `0.6708`
- Direct 12M test AUC: `0.6652`
- Calibration reference: full validation observable 12M loan-level cohort
- Validation required PD from realized 12M loss: `0.0620`; conservative target after 110% buffer: `0.0682`
- Full observable 12M test EL: expected `$499.5M` versus realized `$350.4M`, or `142.5%` coverage.

## Dataset And Temporal Split

- Processed dataset path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/raw/accepted_2007_to_2018Q4.csv`
- Prediction export path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/test_with_pd_best_model.csv`
- Report sample fraction: `1.0`
- Grid search enabled: `True`
- Modeling target: `calendar_time_hazard`

| partition | rows | event_rate_1m | min_snapshot_month | max_snapshot_month | loan_count |
| --- | --- | --- | --- | --- | --- |
| train | 242468 | 0.2500 | 2007-07-01 | 2015-12-01 | 207303 |
| validation | 25000 | 0.1270 | 2016-01-01 | 2016-12-01 | 24725 |
| test | 100000 | 0.2791 | 2017-01-01 | 2018-11-01 | 97245 |

## Cleaning Method

- Keep resolved and active loan states so the calendar-time survival panel can represent events and censored exposure.
- Parse issue_d and last_pymnt_d into monthly dates; performance fields are used only for event/censoring and evaluation.
- Build one row per sample_id and snapshot_month with target_1m equal to next-month charge-off/default.
- Split rows by snapshot_month; the same loan may legitimately appear across train, validation, and test months.
- Convert interest-rate and revolving-utilization percent strings into numeric values.
- Convert invalid DTI, int_rate, and revol_util values to missing before imputation.
- Impute modeled numeric columns using medians fit on train snapshot rows, then apply the same values to validation, test, and scoring snapshots.

## Selected Hazard Features

- Calendar-time inputs include `snapshot_month`, `calendar_year`, `calendar_quarter`, `month_on_book`, `remaining_term`, `seasoning_ratio`, vintage fields, and scheduled-balance proxies.
- Full calendar hazard profile is used for RF/XGBoost/MLP; Logistic uses a compact profile without purpose dummies or high-order interaction columns.
- Enabled main model families are Logistic Regression, Random Forest, XGBoost, and MLP Neural Network; HistGradientBoosting is excluded from the main PD search.
- Champion feature profile: `full_calendar_hazard`.
- Champion modeled column count: `176`.
- In the dual-PD final workflow, `predicted_pd` is the rerun static HGB lifetime PD; `predicted_pd_12m` is the calibrated direct active-snapshot 12-month PD. The calendar-time hazard PD is retained in diagnostic fields such as `predicted_pd_12m_hazard`.

### Encoded Feature Columns

loan_amnt, annual_inc, fico_range_low, dti, emp_length, term_months, int_rate, installment, delinq_2yrs, inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util, total_acc, mort_acc, pub_rec_bankruptcies, month_on_book, remaining_term, seasoning_ratio, calendar_year, vintage_year, scheduled_balance_proxy, scheduled_balance_to_funded, verification_status_Source Verified, verification_status_Verified, home_ownership_NONE, home_ownership_OTHER, home_ownership_OWN, home_ownership_RENT, month_bucket_m04_06, month_bucket_m07_12, month_bucket_m13_24, month_bucket_m25_36, month_bucket_m37_48, month_bucket_m49_60, calendar_quarter_2007Q4, calendar_quarter_2008Q1, calendar_quarter_2008Q2, calendar_quarter_2008Q3, calendar_quarter_2008Q4, calendar_quarter_2009Q1, calendar_quarter_2009Q2, calendar_quarter_2009Q3, calendar_quarter_2009Q4, calendar_quarter_2010Q1, calendar_quarter_2010Q2, calendar_quarter_2010Q3, calendar_quarter_2010Q4, calendar_quarter_2011Q1, calendar_quarter_2011Q2, calendar_quarter_2011Q3, calendar_quarter_2011Q4, calendar_quarter_2012Q1, calendar_quarter_2012Q2, calendar_quarter_2012Q3, calendar_quarter_2012Q4, calendar_quarter_2013Q1, calendar_quarter_2013Q2, calendar_quarter_2013Q3, calendar_quarter_2013Q4, calendar_quarter_2014Q1, calendar_quarter_2014Q2, calendar_quarter_2014Q3, calendar_quarter_2014Q4, calendar_quarter_2015Q1, calendar_quarter_2015Q2, calendar_quarter_2015Q3, calendar_quarter_2015Q4, vintage_quarter_2007Q3, vintage_quarter_2007Q4, vintage_quarter_2008Q1, vintage_quarter_2008Q2, vintage_quarter_2008Q3, vintage_quarter_2008Q4, vintage_quarter_2009Q1, vintage_quarter_2009Q2, vintage_quarter_2009Q3, vintage_quarter_2009Q4, vintage_quarter_2010Q1, vintage_quarter_2010Q2, vintage_quarter_2010Q3, vintage_quarter_2010Q4, vintage_quarter_2011Q1, vintage_quarter_2011Q2, vintage_quarter_2011Q3, vintage_quarter_2011Q4, vintage_quarter_2012Q1, vintage_quarter_2012Q2, vintage_quarter_2012Q3, vintage_quarter_2012Q4, vintage_quarter_2013Q1, vintage_quarter_2013Q2, vintage_quarter_2013Q3, vintage_quarter_2013Q4, vintage_quarter_2014Q1, vintage_quarter_2014Q2, vintage_quarter_2014Q3, vintage_quarter_2014Q4, vintage_quarter_2015Q1, vintage_quarter_2015Q2, vintage_quarter_2015Q3, vintage_quarter_2015Q4, purpose_credit_card, purpose_debt_consolidation, purpose_educational, purpose_home_improvement, purpose_house, purpose_major_purchase, purpose_medical, purpose_moving, purpose_other, purpose_renewable_energy, purpose_small_business, purpose_vacation, purpose_wedding, term_month_interaction_36_m04_06, term_month_interaction_36_m07_12, term_month_interaction_36_m13_24, term_month_interaction_36_m25_36, term_month_interaction_60_m01_03, term_month_interaction_60_m04_06, term_month_interaction_60_m07_12, term_month_interaction_60_m13_24, term_month_interaction_60_m25_36, term_month_interaction_60_m37_48, term_month_interaction_60_m49_60, purpose_group_month_bucket_credit_card_m04_06, purpose_group_month_bucket_credit_card_m07_12, purpose_group_month_bucket_credit_card_m13_24, purpose_group_month_bucket_credit_card_m25_36, purpose_group_month_bucket_credit_card_m37_48, purpose_group_month_bucket_credit_card_m49_60, purpose_group_month_bucket_debt_consolidation_m01_03, purpose_group_month_bucket_debt_consolidation_m04_06, purpose_group_month_bucket_debt_consolidation_m07_12, purpose_group_month_bucket_debt_consolidation_m13_24, purpose_group_month_bucket_debt_consolidation_m25_36, purpose_group_month_bucket_debt_consolidation_m37_48, purpose_group_month_bucket_debt_consolidation_m49_60, purpose_group_month_bucket_home_improvement_m01_03, purpose_group_month_bucket_home_improvement_m04_06, purpose_group_month_bucket_home_improvement_m07_12, purpose_group_month_bucket_home_improvement_m13_24, purpose_group_month_bucket_home_improvement_m25_36, purpose_group_month_bucket_home_improvement_m37_48, purpose_group_month_bucket_home_improvement_m49_60, purpose_group_month_bucket_major_purchase_m01_03, purpose_group_month_bucket_major_purchase_m04_06, purpose_group_month_bucket_major_purchase_m07_12, purpose_group_month_bucket_major_purchase_m13_24, purpose_group_month_bucket_major_purchase_m25_36, purpose_group_month_bucket_major_purchase_m37_48, purpose_group_month_bucket_major_purchase_m49_60, purpose_group_month_bucket_other_m01_03, purpose_group_month_bucket_other_m04_06, purpose_group_month_bucket_other_m07_12, purpose_group_month_bucket_other_m13_24, purpose_group_month_bucket_other_m25_36, purpose_group_month_bucket_other_m37_48, purpose_group_month_bucket_other_m49_60, fico_bucket_month_bucket_unknown_m04_06, fico_bucket_month_bucket_unknown_m07_12, fico_bucket_month_bucket_unknown_m13_24, fico_bucket_month_bucket_unknown_m25_36, fico_bucket_month_bucket_unknown_m37_48, fico_bucket_month_bucket_unknown_m49_60

## Candidate Models And Search Space

- Enabled model families: `logistic_regression, random_forest, xgboost, mlp_neural_network`
- Grid search models: `random_forest, xgboost`

### logistic_regression

Base parameters:

```json
{
  "C": 1.0,
  "max_iter": 300,
  "class_weight": null
}
```

Grid search space:

```json
{}
```

### random_forest

Base parameters:

```json
{
  "n_estimators": 80,
  "max_depth": 10,
  "min_samples_split": 200,
  "min_samples_leaf": 50,
  "max_features": "sqrt",
  "n_jobs": -1,
  "random_state": 42,
  "bootstrap": true,
  "max_samples": 0.7
}
```

Grid search space:

```json
{}
```

### xgboost

Base parameters:

```json
{
  "n_estimators": 120,
  "max_depth": 4,
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
{}
```

### mlp_neural_network

Base parameters:

```json
{
  "activation": "relu",
  "solver": "adam",
  "max_iter": 30,
  "early_stopping": true,
  "validation_fraction": 0.05,
  "n_iter_no_change": 3,
  "random_state": 42,
  "hidden_layer_sizes": [
    64,
    32,
    16
  ],
  "batch_size": 4096,
  "alpha": 0.0001,
  "learning_rate_init": 0.001
}
```

Grid search space:

```json
{}
```

## Validation Ranking

| validation_rank | model_name | feature_profile | feature_count | search_mode | search_candidates | params | validation_auc | validation_ks | validation_brier | validation_12m_auc | validation_12m_brier | calibration_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost | full_calendar_hazard | 167 | 30min_fast_1m_hazard_grid | 4 | {"colsample_bytree": 0.8, "learning_rate": 0.08, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7138 | 0.3085 | 0.1208 | 0.7008 | 0.6513 | Overestimates risk in the highest-PD bins. |
| 2 | Random Forest | full_calendar_hazard | 167 | 30min_fast_1m_hazard_grid | 3 | {"bootstrap": true, "max_depth": 12, "max_features": "sqrt", "max_samples": 0.7, "min_samples_leaf": 50, "min_samples_split": 200, "n_estimators": 90, "n_jobs": -1, "random_state": 42} | 0.7023 | 0.2974 | 0.1198 | 0.6905 | 0.7213 | Visible calibration drift; inspect the curve before finalizing. |
| 3 | MLP Neural Network | full_calendar_hazard | 167 | 30min_fast_1m_hazard_grid | 1 | {"activation": "relu", "alpha": 0.0001, "batch_size": 4096, "early_stopping": true, "hidden_layer_sizes": [64, 32, 16], "learning_rate_init": 0.001, "max_iter": 30, "n_iter_no_change": 3, "random_state": 42, "solver": "adam", "validation_fraction": 0.05} | 0.7026 | 0.2947 | 0.1231 | NA | NA | Overestimates risk in the highest-PD bins. |
| 4 | Logistic Regression | compact_calendar_hazard | 103 | 30min_fast_1m_hazard_grid | 1 | {"C": 1.0, "class_weight": null, "max_iter": 300} | 0.6990 | 0.2963 | 0.1511 | NA | NA | Overestimates risk in the highest-PD bins. |

## Validation Visual Diagnostics

Calibration markers are sized by bin count so sparse tail bins are visually distinguishable from dense central bins.

### Observed Default Rate Trend

![Observed Default Rate Trend](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/default_rate_trend.png)

### Validation ROC Comparison

![Validation ROC Comparison](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/validation_roc_comparison.png)

### Validation Reliability Diagram

![Validation Reliability Diagram](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/validation_calibration_comparison.png)

## Grid Search Results

The table below shows the top 10 validation-ranked candidates per model family out of `9` total grid-search candidates.

| model_name | search_rank | params | validation_auc | validation_ks | validation_brier |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1 | {"C": 1.0, "class_weight": null, "max_iter": 300} | 0.6990 | 0.2963 | 0.1511 |
| Random Forest | 2 | {"bootstrap": true, "max_depth": 8, "max_features": "sqrt", "max_samples": 0.6, "min_samples_leaf": 75, "min_samples_split": 200, "n_estimators": 60, "n_jobs": -1, "random_state": 42} | 0.6931 | 0.2854 | 0.1210 |
| Random Forest | 3 | {"bootstrap": true, "max_depth": 10, "max_features": "sqrt", "max_samples": 0.7, "min_samples_leaf": 50, "min_samples_split": 200, "n_estimators": 80, "n_jobs": -1, "random_state": 42} | 0.6993 | 0.2944 | 0.1200 |
| Random Forest | 4 | {"bootstrap": true, "max_depth": 12, "max_features": "sqrt", "max_samples": 0.7, "min_samples_leaf": 50, "min_samples_split": 200, "n_estimators": 90, "n_jobs": -1, "random_state": 42} | 0.7023 | 0.2974 | 0.1198 |
| XGBoost | 5 | {"colsample_bytree": 0.8, "learning_rate": 0.08, "max_depth": 4, "min_child_weight": 5, "n_estimators": 80, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7092 | 0.3010 | 0.1195 |
| XGBoost | 6 | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 80, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7104 | 0.3049 | 0.1195 |
| XGBoost | 7 | {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7091 | 0.3026 | 0.1194 |
| XGBoost | 8 | {"colsample_bytree": 0.8, "learning_rate": 0.08, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7138 | 0.3085 | 0.1208 |
| MLP Neural Network | 9 | {"activation": "relu", "alpha": 0.0001, "batch_size": 4096, "early_stopping": true, "hidden_layer_sizes": [64, 32, 16], "learning_rate_init": 0.001, "max_iter": 30, "n_iter_no_change": 3, "random_state": 42, "solver": "adam", "validation_fraction": 0.05} | 0.7026 | 0.2947 | 0.1231 |

## Final Chosen Hyperparameters

- Selected model: `XGBoost`
- Validation AUC: `0.7138`
- Validation KS: `0.3085`
- Validation Brier: `0.1208`

```json
{"colsample_bytree": 0.8, "learning_rate": 0.08, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8}
```

## Holdout Test Results

- Champion test AUC: `0.7011`
- Champion test KS: `0.2936`
- Champion test Brier: `0.2020`

| validation_rank | model_name | feature_profile | feature_count | params | test_auc | test_ks | test_brier | test_12m_auc | test_12m_brier | calibration_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost | full_calendar_hazard | 176 | {"colsample_bytree": 0.8, "learning_rate": 0.08, "max_depth": 6, "min_child_weight": 5, "n_estimators": 120, "n_jobs": -1, "random_state": 42, "reg_lambda": 1.0, "subsample": 0.8} | 0.7011 | 0.2936 | 0.2020 | 0.6837 | 0.4078 | Underestimates risk in the highest-PD bins. |

## Champion Temporal Performance

| period | fitted_on | rows | default_rate | min_snapshot_month | max_snapshot_month | auc | ks | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | train only | 242468 | 0.2500 | 2007-07-01 | 2015-12-01 | 0.7376 | 0.3458 | 0.1631 |
| validation | train only | 25000 | 0.1270 | 2016-01-01 | 2016-12-01 | 0.7138 | 0.3085 | 0.1208 |
| test | train + validation | 100000 | 0.2791 | 2017-01-01 | 2018-11-01 | 0.7011 | 0.2936 | 0.2020 |

## Holdout Visual Diagnostics

### Test ROC Comparison

![Test ROC Comparison](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/test_roc_comparison.png)

### Test Reliability Diagram

![Test Reliability Diagram](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/test_calibration_comparison.png)

### Champion Validation KS Curve

![Champion Validation KS Curve](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/champion_validation_ks.png)

### Champion Test KS Curve

![Champion Test KS Curve](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/champion_test_ks.png)

### Champion Score Distribution

![Champion Score Distribution](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/champion_score_distribution.png)

## Feature Importance And Interpretability

### Logistic Regression Coefficients

Logistic regression coefficients were not available.

### XGBoost Feature Importance

| feature | importance |
| --- | --- |
| int_rate | 0.0949 |
| purpose_group_month_bucket_debt_consolidation_m01_03 | 0.0732 |
| seasoning_ratio | 0.0501 |
| month_on_book | 0.0409 |
| calendar_year | 0.0248 |
| calendar_quarter_2016Q2 | 0.0185 |
| calendar_quarter_2015Q3 | 0.0158 |
| inq_last_6mths | 0.0152 |
| calendar_quarter_2016Q4 | 0.0150 |
| scheduled_balance_to_funded | 0.0130 |
| annual_inc | 0.0114 |
| remaining_term | 0.0112 |
| home_ownership_RENT | 0.0108 |
| purpose_small_business | 0.0105 |
| calendar_quarter_2014Q4 | 0.0103 |

## Visual Interpretability

### XGBoost Feature Importance

![XGBoost Feature Importance](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/temporal_model/xgboost_feature_importance.png)

## Candidate Feature Flags

- `use_additional_numeric_features`: Additional origination-time numeric credit features
- `use_extended_engineered_features`: Extended engineered ratios and delinquency flags
- `use_verification_status_dummies`: Verification status dummies
- `use_grade_dummies`: Grade dummies
- `use_initial_list_status_dummies`: Initial list status dummies
- `use_application_type_dummies`: Application type dummies

## 30-Minute Training Run Notes

- This run reuses the existing calendar-time modeling sample and does not rebuild the raw Dask sample.
- Train/validation/test caps: `250,000` / `25,000` / `100,000`.
- Grid search uses fast 1-month hazard scoring; lifetime and 12-month PD aggregation is limited to RF/XGB model-best candidates and the final champion.
- Planned candidates: `9`; ran candidates: `9`.
- Total training/report runtime before downstream loss workflow: `440.7` seconds.
- Selected champion: `xgboost`.
