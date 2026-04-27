# Credit Loss, CECL Proxy, And Risk Segmentation Report

## Course Formula Alignment

This phase extends the hazard PD workflow into a course-aligned expected-loss framework.
The core lens is `Expected Loss = PD x LGD x EAD`, with installment-loan simplifications for EAD, credibility-weighted LGD lookups, a rerun original static HGB model for lifetime PD, a direct active-snapshot XGBoost model for twelve-month PD, and management-facing segmentation.
Pricing and economic capital are not implemented here; the report ends with a short bridge showing how these outputs feed those later modules.

## Data Assets And Temporal Policy

- Main PD prediction path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/test_with_pd_best_model.csv`
- Loss workflow dataset path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/loss_workflow_dataset.csv`
- Charged-off LGD/EAD proxy path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/charged_off_loss_proxy.csv`
- Active reserve snapshot path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/cecl_active_snapshot.csv`
- Holdout loss evaluation path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/test_with_loss_metrics.csv`
- Segment summary path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/segment_expected_loss_summary.csv`

The temporal split remains origination-based throughout the workflow: train vintages before 2016, validation vintages in 2016, and test vintages from 2017 onward.

### Workflow Dataset Summary

| split_label | loan_status | loan_count | funded_amount | avg_months_on_book |
| --- | --- | --- | --- | --- |
| test | Charged Off | 48015 | 769827075.0 | 8.933080188578622 |
| test | Current | 689032 | 10847888775.0 | 12.457195021421356 |
| test | Default | 28 | 410575.0 | 17.0 |
| test | Fully Paid | 177596 | 2489341750.0 | 9.423776436406225 |
| test | In Grace Period | 5830 | 102118275.0 | 13.962264150943396 |
| test | Late (16-30 days) | 3095 | 53463175.0 | 13.183521809369951 |
| test | Late (31-120 days) | 15225 | 258170600.0 | 12.735568339615156 |
| train | Charged Off | 152302 | 2356932200.0 | 18.8293371317032 |
| train | Current | 55224 | 1112566075.0 | 45.99581703607127 |
| train | Default | 2 | 47000.0 | 47.0 |
| train | Fully Paid | 674302 | 9531665075.0 | 26.310630844932984 |
| train | In Grace Period | 813 | 16499750.0 | 45.77613776137761 |
| train | Late (16-30 days) | 352 | 7221350.0 | 45.12215909090909 |
| train | Late (31-120 days) | 1696 | 33222800.0 | 43.558372641509436 |
| validation | Charged Off | 68242 | 1051229150.0 | 14.763986710963454 |
| validation | Current | 134061 | 2042363400.0 | 31.530325747234468 |
| validation | Default | 10 | 116450.0 | 29.5 |
| validation | Fully Paid | 224853 | 3189016000.0 | 19.221526953164958 |
| validation | In Grace Period | 1793 | 30467675.0 | 31.496932515337424 |
| validation | Late (16-30 days) | 902 | 14949450.0 | 31.093126385809313 |
| validation | Late (31-120 days) | 4546 | 72399575.0 | 29.844478662560494 |

## Cleaning And Proxy Construction

- Load origination and performance fields required for LGD, EAD, EL, and reserve analytics.
- Parse issue_d and last_pymnt_d into issue_date and last_payment_date.
- Keep resolved, current, grace-period, late, and default statuses needed for event and censoring logic.
- Convert funded amount, balances, payments, recoveries, installment, income, and FICO fields to numeric.
- Parse term into term_months and interest rate into a numeric annual percentage.
- Cap months_on_book at term_months for amortization-aware reserve analytics.
- Map purpose into management buckets and annual income into reporting bands.
- Use temporal split labels based on origination date so future vintages never enter training.

### LGD And EAD Proxy Design

- `EAD_proxy = max(funded_amnt - total_rec_prncp, 0)` for charged-off loans.
- `net_recoveries = max(recoveries - collection_recovery_fee, 0)`.
- `LGD_proxy = clip(1 - net_recoveries / EAD_proxy, 0, 1)` when `EAD_proxy > 0`.
- `EAD_current` uses `out_prncp` when available and falls back to a scheduled-balance amortization proxy.

- Charged-off proxy rows kept: `268537`
- Charged-off proxy portfolio LGD mean: `0.9086`
- Charged-off proxy average EAD: `11170.0973`

### Expected LGD Lookup

- Fit splits: `train, validation`
- Shrinkage rule: `weight = n / (n + 50)`
- Portfolio fallback LGD: `0.9014`
- Charged-off training rows: `220529`
- Holdout charged-off LGD MAE: `0.0692`
- Holdout charged-off LGD RMSE: `0.0892`

Lookup-source usage on the holdout and reserve outputs:

| lgd_lookup_source | loan_count | analysis_scope |
| --- | --- | --- |
| grade_term | 225611 | resolved_test |
| exact_segment | 199399 | active_snapshot |
| grade_term | 713210 | active_snapshot |

## Main PD Input And Auxiliary Reserve Hazard Benchmark

The main expected-loss input is the dual-PD stage2 file: static HistGradientBoosting supplies lifetime PD and calibrated direct active-snapshot XGBoost supplies twelve-month PD. The auxiliary reserve hazard benchmark below is retained only for diagnostics and comparison.

- Main PD champion used for EL: `Dual PD: Static HGB lifetime + direct active-snapshot 12M`
- Prediction source: `precomputed_dual_pd_static_lifetime_and_direct_active_snapshot_12m`
- Full-stage2 scoring: `True`
- Unscored rows fall back to reserve hazard PD: `False`

The table below reports the downstream reserve auxiliary hazard benchmark. It remains useful as a time-consistent diagnostic, but main portfolio EL uses the dual-PD stage2 predictions when available.

- Selected auxiliary reserve hazard model: `HistGradientBoosting Hazard`
- Auxiliary reserve hazard parameters: `{'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.01, 'learning_rate': 0.03, 'max_depth': 6, 'max_iter': 180, 'min_samples_leaf': 120}`
- Auxiliary hazard full grid output: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/hazard_grid_search_results.csv`
- Auxiliary hazard comparison output: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/hazard_model_comparison.csv`

### Hazard Panel Summary

| split_label | panel_cells | exposure_count | event_count |
| --- | --- | --- | --- |
| test | 3755 | 10996348 | 46877 |
| train | 4590 | 23264938 | 151644 |
| validation | 4094 | 9771937 | 67725 |

### Hazard Model Comparison

| model_key | model_name | selected | params | validation_lifetime_auc | validation_lifetime_ks | validation_lifetime_brier | validation_12m_auc | validation_12m_brier | test_lifetime_auc | test_lifetime_ks | test_lifetime_brier | test_12m_auc | test_12m_brier | validation_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | HistGradientBoosting Hazard | True | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.01, 'learning_rate': 0.03, 'max_depth': 6, 'max_iter': 180, 'min_samples_leaf': 120} | 0.6944 | 0.2800 | 0.1629 | 0.6942 | 0.0890 | 0.6833 | 0.2706 | 0.1551 | 0.6712 | 0.1375 | 1 |
| pooled_logistic | Pooled Logistic Hazard | False | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 0.25, 'class_weight': None} | 0.6943 | 0.2797 | 0.1628 | 0.6939 | 0.0892 | 0.6830 | 0.2685 | 0.1554 | 0.6709 | 0.1380 | 2 |

### Hazard Parameter Search

The table below shows the top 20 of `58` validation-ranked hazard candidates; the full candidate-level output is written to `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/hazard_grid_search_results.csv`.

| candidate_id | model_key | model_name | params | validation_lifetime_auc | validation_lifetime_ks | validation_lifetime_brier | validation_12m_auc | validation_12m_brier | validation_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.01, 'learning_rate': 0.03, 'max_depth': 6, 'max_iter': 180, 'min_samples_leaf': 120} | 0.6944 | 0.2800 | 0.1629 | 0.6942 | 0.0890 | 1 |
| 18 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.0, 'learning_rate': 0.03, 'max_depth': 6, 'max_iter': 180, 'min_samples_leaf': 120} | 0.6944 | 0.2800 | 0.1629 | 0.6942 | 0.0890 | 2 |
| 54 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.01, 'learning_rate': 0.08, 'max_depth': 4, 'max_iter': 180, 'min_samples_leaf': 120} | 0.6943 | 0.2798 | 0.1629 | 0.6930 | 0.0889 | 3 |
| 30 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.0, 'learning_rate': 0.08, 'max_depth': 4, 'max_iter': 180, 'min_samples_leaf': 120} | 0.6943 | 0.2798 | 0.1629 | 0.6930 | 0.0889 | 4 |
| 53 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.01, 'learning_rate': 0.08, 'max_depth': 4, 'max_iter': 180, 'min_samples_leaf': 80} | 0.6943 | 0.2799 | 0.1629 | 0.6933 | 0.0888 | 5 |
| 29 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.0, 'learning_rate': 0.08, 'max_depth': 4, 'max_iter': 180, 'min_samples_leaf': 80} | 0.6943 | 0.2799 | 0.1629 | 0.6933 | 0.0888 | 6 |
| 1 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 0.25, 'class_weight': None} | 0.6943 | 0.2797 | 0.1628 | 0.6939 | 0.0892 | 7 |
| 2 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 0.25, 'class_weight': 'balanced'} | 0.6943 | 0.2797 | 0.1674 | 0.6939 | 0.0873 | 8 |
| 3 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 0.5, 'class_weight': None} | 0.6943 | 0.2797 | 0.1628 | 0.6939 | 0.0892 | 9 |
| 6 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 1.0, 'class_weight': 'balanced'} | 0.6943 | 0.2798 | 0.1674 | 0.6939 | 0.0873 | 10 |
| 10 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 5.0, 'class_weight': 'balanced'} | 0.6943 | 0.2798 | 0.1674 | 0.6939 | 0.0873 | 11 |
| 8 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 2.0, 'class_weight': 'balanced'} | 0.6943 | 0.2797 | 0.1674 | 0.6939 | 0.0873 | 12 |
| 7 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 2.0, 'class_weight': None} | 0.6943 | 0.2797 | 0.1628 | 0.6939 | 0.0892 | 13 |
| 5 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 1.0, 'class_weight': None} | 0.6943 | 0.2797 | 0.1628 | 0.6939 | 0.0892 | 14 |
| 9 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 5.0, 'class_weight': None} | 0.6943 | 0.2797 | 0.1628 | 0.6939 | 0.0892 | 15 |
| 4 | pooled_logistic | Pooled Logistic Hazard | {'solver': 'lbfgs', 'max_iter': 500, 'n_jobs': None, 'C': 0.5, 'class_weight': 'balanced'} | 0.6943 | 0.2797 | 0.1674 | 0.6939 | 0.0873 | 16 |
| 48 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.01, 'learning_rate': 0.05, 'max_depth': 6, 'max_iter': 120, 'min_samples_leaf': 120} | 0.6943 | 0.2797 | 0.1629 | 0.6944 | 0.0889 | 17 |
| 24 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.0, 'learning_rate': 0.05, 'max_depth': 6, 'max_iter': 120, 'min_samples_leaf': 120} | 0.6943 | 0.2797 | 0.1629 | 0.6944 | 0.0889 | 18 |
| 50 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.01, 'learning_rate': 0.05, 'max_depth': 6, 'max_iter': 180, 'min_samples_leaf': 120} | 0.6942 | 0.2790 | 0.1629 | 0.6942 | 0.0889 | 19 |
| 26 | hist_gradient_boosting | HistGradientBoosting Hazard | {'max_leaf_nodes': 31, 'early_stopping': False, 'random_state': 42, 'l2_regularization': 0.0, 'learning_rate': 0.05, 'max_depth': 6, 'max_iter': 180, 'min_samples_leaf': 120} | 0.6942 | 0.2790 | 0.1629 | 0.6942 | 0.0889 | 20 |

### Validation Hazard Metrics

| split_name | hazard_model_name | pd_measure | auc | ks | brier |
| --- | --- | --- | --- | --- | --- |
| validation | HistGradientBoosting Hazard | lifetime | 0.6944 | 0.2800 | 0.1629 |
| validation | HistGradientBoosting Hazard | 12m | 0.6942 | 0.2821 | 0.0890 |

### Test Hazard Metrics

| split_name | hazard_model_name | pd_measure | auc | ks | brier |
| --- | --- | --- | --- | --- | --- |
| test | HistGradientBoosting Hazard | lifetime | 0.6833 | 0.2706 | 0.1551 |
| test | HistGradientBoosting Hazard | 12m | 0.6712 | 0.2557 | 0.1375 |

### Validation Calibration By Vintage

| issue_year | loan_count | actual_default_rate | predicted_lifetime_pd | actual_default_12m_rate | predicted_12m_pd | split_name |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 292578 | 0.2315 | 0.2297 | 0.1005 | 0.0589 | validation |

### Test Calibration By Vintage

| issue_year | loan_count | actual_default_rate | predicted_lifetime_pd | actual_default_12m_rate | predicted_12m_pd | split_name |
| --- | --- | --- | --- | --- | --- | --- |
| 2017 | 168700 | 0.2285 | 0.2390 | 0.1616 | 0.0657 | test |
| 2018 | 55773 | 0.1493 | 0.2230 | 0.1490 | 0.0599 | test |

## Expected Loss Outputs

| analysis_scope | pd_measure | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | portfolio_el_share | actual_loss_amount | actual_loss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observable_12m_test | 12m | 443579 | 6584957075.0000 | 0.0795 | 0.9017 | 14845.0605 | 1126.0841 | 499507241.2874 | 1.0000 | 350447341.16339725 | 0.0532 |
| resolved_test | 12m | 225611 | 3259168825.0000 | 0.0823 | 0.9011 | 14445.9660 | 1144.0733 | 258115528.0035 | 1.0000 | 477277111.178897 | 0.1464 |
| resolved_test | lifetime | 225611 | 3259168825.0000 | 0.2613 | 0.9011 | 14445.9660 | 3725.6724 | 840552674.1255 | 1.0000 | 616265196.9735968 | 0.1891 |
| active_snapshot | 12m | 912609 | 14591904925.0000 | 0.0856 | 0.9048 | 10467.6862 | 851.6798 | 777250645.0214 | 1.0000 |  | NA |
| active_snapshot | lifetime | 912609 | 14591904925.0000 | 0.2613 | 0.9048 | 10467.6862 | 2690.9032 | 2455742436.2835 | 1.0000 |  | NA |

### Fixed-Horizon PD Input For 12-Month EL

- Locked dual-PD champion: `Dual PD: Static HGB lifetime + direct active-snapshot 12M`
- Prediction source: `precomputed_dual_pd_static_lifetime_and_direct_active_snapshot_12m`
- Rows scored with the fixed-horizon model: `1138220`
- Mean compatibility `predicted_pd` across scored rows, interpreted as lifetime PD: `0.2605`
- Mean 12-month PD across scored rows: `0.0834`
- Mean lifetime PD across scored rows: `0.2605`
- Used as the 12-month EL PD input: `True`
- Stage2 scoring cap per scope: `None`
- Resolved rows scored by stage2: `225611`
- Active rows scored by stage2: `912609`
- Unscored rows fall back to reserve hazard PD: `False`
- Twelve-month PD model source: `direct_active_snapshot_12m_xgboost`
- Twelve-month calibration method: `loss_weighted_logit_intercept_plus_grade_term_floor`
- Twelve-month calibration intercept shift: `-1.3518`
- Twelve-month calibration reference required PD: `0.0620`
- Twelve-month calibration target PD with buffer: `0.0682`
- Conservative 12-month overlay method: `validation_grade_term_conservative_floor`
- Conservative overlay buffer: `1.1000`
- Conservative overlay cap: `static_lifetime_pd`
- Validation portfolio floor PD before buffer: `0.0628`

### Holdout Loan-Level Examples

| sample_id | issue_date | loan_status | stage2_model_name | auxiliary_hazard_model_name | stage2_champion_pd | pd_12m_fixed_horizon | pd_12m_hazard | lifetime_pd_hazard | expected_lgd | ead_reference | expected_loss_12m | expected_loss_lifetime | actual_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1718632 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.1177 | 0.0585 | 0.0259 | 0.1177 | 0.9003 | 15000.0000 | 790.2505 | 1589.5222 | 0.0000 |
| 1718640 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.3103 | 0.0648 | 0.0677 | 0.3103 | 0.8983 | 21000.0000 | 1221.7697 | 5854.3187 | 0.0000 |
| 1718647 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.1747 | 0.0648 | 0.0653 | 0.1747 | 0.8983 | 17000.0000 | 989.0517 | 2667.3761 | 0.0000 |
| 1718651 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.4124 | 0.1605 | 0.1223 | 0.4124 | 0.9040 | 30000.0000 | 4351.4359 | 11183.0828 | 0.0000 |
| 1718653 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.2580 | 0.1103 | 0.1030 | 0.2580 | 0.8969 | 2000.0000 | 197.9250 | 462.8328 | 0.0000 |
| 1718655 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.2852 | 0.0648 | 0.0798 | 0.2852 | 0.8983 | 10000.0000 | 581.7951 | 2561.6496 | 0.0000 |
| 1718665 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.2964 | 0.0855 | 0.0588 | 0.2964 | 0.9085 | 21600.0000 | 1677.9805 | 5817.2047 | 0.0000 |
| 1718669 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.1442 | 0.0648 | 0.0658 | 0.1442 | 0.8983 | 15000.0000 | 872.6926 | 1942.4473 | 0.0000 |
| 1718670 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.1170 | 0.0585 | 0.0353 | 0.1170 | 0.9000 | 4000.0000 | 210.6738 | 421.1283 | 0.0000 |
| 1718672 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.0611 | 0.0585 | 0.0259 | 0.0611 | 0.9003 | 6000.0000 | 316.1002 | 330.0256 | 0.0000 |

## CECL Proxy And Reserve Snapshot

The reserve view combines lifetime PD from the main stage2 champion (Dual PD: Static HGB lifetime + direct active-snapshot 12M) with credibility-weighted expected LGD and current EAD.

### Active Snapshot Examples

| sample_id | issue_date | loan_status | remaining_term | stage2_model_name | auxiliary_hazard_model_name | stage2_champion_pd | pd_12m_fixed_horizon | pd_12m_hazard | lifetime_pd_hazard | expected_lgd | ead_current | expected_loss_12m | lifetime_expected_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1781187 | 2013-10-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.4430 | 0.1103 | 0.0000 | 0.4430 | 0.8851 | 125.0500 | 12.2119 | 49.0362 |
| 1790999 | 2013-10-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.4130 | 0.0855 | 0.0000 | 0.4130 | 0.8882 | 13730.7023 | 1042.7683 | 5036.2104 |
| 1751337 | 2013-12-01 00:00:00 | Late (31-120 days) | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.5229 | 0.0855 | 0.0000 | 0.5229 | 0.8882 | 11.1900 | 0.8498 | 5.1972 |
| 1753200 | 2013-12-01 00:00:00 | Late (31-120 days) | 1.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.4982 | 0.2261 | 0.0096 | 0.4982 | 0.8823 | 286.3600 | 57.1140 | 125.8665 |
| 1754434 | 2013-12-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.3452 | 0.1103 | 0.0000 | 0.3452 | 0.8851 | 1615.5400 | 157.7677 | 493.6460 |
| 1760475 | 2013-12-01 00:00:00 | In Grace Period | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.4064 | 0.0855 | 0.0000 | 0.4064 | 0.8882 | 553.5300 | 42.0374 | 199.8001 |
| 1761504 | 2013-12-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.3716 | 0.1103 | 0.0000 | 0.3716 | 0.8851 | 931.3100 | 90.9483 | 306.2866 |
| 1761719 | 2013-12-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.1843 | 0.0855 | 0.0000 | 0.1843 | 0.8882 | 10314.2289 | 783.3067 | 1688.4701 |
| 1761741 | 2013-12-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.7042 | 0.2823 | 0.0000 | 0.7042 | 0.8777 | 507.2800 | 125.6823 | 313.5151 |
| 1763190 | 2013-12-01 00:00:00 | Late (31-120 days) | 0.0 | Dual PD: Static HGB lifetime + direct active-snapshot 12M | HistGradientBoosting Hazard | 0.2372 | 0.0855 | 0.0000 | 0.2372 | 0.8878 | 803.5500 | 61.0032 | 169.2398 |

## Risk Segmentation Analysis

The management-facing cuts in this phase are `grade, fico_bucket, term_months, issue_year, issue_year_quarter, purpose_group, annual_income_band`.

### Top Segment Contributions: observable_12m_test

| segment_value | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | segment_type | analysis_scope | pd_measure | portfolio_el_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50-100k | 217236 | 3162039350.0000 | 0.0798 | 0.9019 | 14555.7797 | 1129.0570 | 245271820.8757 | annual_income_band | observable_12m_test | 12m | 0.4910 |
| 100-150k | 68800 | 1368726450.0000 | 0.0780 | 0.9028 | 19894.2798 | 1472.3678 | 101298907.6436 | annual_income_band | observable_12m_test | 12m | 0.2028 |
| <50k | 122592 | 1201104450.0000 | 0.0812 | 0.9004 | 9797.5761 | 769.9619 | 94391169.0842 | annual_income_band | observable_12m_test | 12m | 0.1890 |
| 150k+ | 34951 | 853086825.0000 | 0.0747 | 0.9030 | 24408.0806 | 1675.0692 | 58545343.6840 | annual_income_band | observable_12m_test | 12m | 0.1172 |
| fair | 246250 | 3438891625.0000 | 0.0859 | 0.9009 | 13965.0421 | 1154.6759 | 284338948.4160 | fico_bucket | observable_12m_test | 12m | 0.5692 |
| good | 135709 | 2176951125.0000 | 0.0755 | 0.9024 | 16041.3173 | 1149.9907 | 156064088.5408 | fico_bucket | observable_12m_test | 12m | 0.3124 |
| very_good | 61620 | 969114325.0000 | 0.0630 | 0.9033 | 15727.2691 | 959.1724 | 59104204.3306 | fico_bucket | observable_12m_test | 12m | 0.1183 |
| C | 145144 | 2224923850.0000 | 0.0731 | 0.9023 | 15329.0791 | 1055.0900 | 153139983.5521 | grade | observable_12m_test | 12m | 0.3066 |
| B | 133127 | 1839662450.0000 | 0.0634 | 0.9024 | 13818.8531 | 830.6319 | 110579535.0289 | grade | observable_12m_test | 12m | 0.2214 |
| D | 56646 | 891886825.0000 | 0.1105 | 0.9007 | 15744.9215 | 1570.1812 | 88944486.1979 | grade | observable_12m_test | 12m | 0.1781 |
| A | 78796 | 1096915000.0000 | 0.0545 | 0.9012 | 13920.9478 | 691.3250 | 54473642.6121 | grade | observable_12m_test | 12m | 0.1091 |
| E | 20167 | 340651725.0000 | 0.1622 | 0.8994 | 16891.5419 | 2472.7862 | 49868680.2618 | grade | observable_12m_test | 12m | 0.0998 |
| 2017 | 443579 | 6584957075.0000 | 0.0795 | 0.9017 | 14845.0605 | 1126.0841 | 499507241.2874 | issue_year | observable_12m_test | 12m | 1.0000 |
| 2017Q3 | 122701 | 1791201400.0000 | 0.0805 | 0.9017 | 14598.0994 | 1128.9228 | 138519956.8427 | issue_year_quarter | observable_12m_test | 12m | 0.2773 |
| 2017Q4 | 118648 | 1817354125.0000 | 0.0785 | 0.9022 | 15317.1914 | 1137.7085 | 134986837.5744 | issue_year_quarter | observable_12m_test | 12m | 0.2702 |
| 2017Q2 | 105451 | 1538432075.0000 | 0.0793 | 0.9016 | 14589.0705 | 1107.7100 | 116809131.3872 | issue_year_quarter | observable_12m_test | 12m | 0.2338 |
| 2017Q1 | 96779 | 1437969475.0000 | 0.0797 | 0.9013 | 14858.2799 | 1128.2542 | 109191315.4832 | issue_year_quarter | observable_12m_test | 12m | 0.2186 |
| debt_consolidation | 245083 | 3926847225.0000 | 0.0821 | 0.9018 | 16022.5198 | 1249.4183 | 306211189.6915 | purpose_group | observable_12m_test | 12m | 0.6130 |
| credit_card | 91466 | 1364811275.0000 | 0.0728 | 0.9019 | 14921.5148 | 1032.2601 | 94416698.5958 | purpose_group | observable_12m_test | 12m | 0.1890 |
| other | 61197 | 642544250.0000 | 0.0810 | 0.9005 | 10499.6037 | 825.9357 | 50544788.2724 | purpose_group | observable_12m_test | 12m | 0.1012 |
| home_improvement | 34693 | 506094600.0000 | 0.0762 | 0.9021 | 14587.8016 | 1068.1201 | 37056289.4298 | purpose_group | observable_12m_test | 12m | 0.0742 |
| major_purchase | 11140 | 144659725.0000 | 0.0788 | 0.9015 | 12985.6127 | 1012.4125 | 11278275.2980 | purpose_group | observable_12m_test | 12m | 0.0226 |
| 36 | 320419 | 3994539275.0000 | 0.0694 | 0.8991 | 12466.6118 | 783.9662 | 251197669.5877 | term_months | observable_12m_test | 12m | 0.5029 |
| 60 | 123160 | 2590417800.0000 | 0.1059 | 0.9084 | 21032.9474 | 2016.1544 | 248309571.6997 | term_months | observable_12m_test | 12m | 0.4971 |

### Top Segment Contributions: resolved_test

| segment_value | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | segment_type | analysis_scope | pd_measure | portfolio_el_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50-100k | 112648 | 1600802175.0000 | 0.0826 | 0.9013 | 14210.6578 | 1150.3862 | 129588703.8099 | annual_income_band | resolved_test | 12m | 0.5021 |
| 100-150k | 35064 | 676464750.0000 | 0.0808 | 0.9020 | 19292.2870 | 1490.1939 | 52252158.9117 | annual_income_band | resolved_test | 12m | 0.2024 |
| <50k | 61345 | 594303200.0000 | 0.0841 | 0.9001 | 9687.8833 | 794.6078 | 48745216.9187 | annual_income_band | resolved_test | 12m | 0.1889 |
| 150k+ | 16554 | 387598700.0000 | 0.0771 | 0.9023 | 23414.2020 | 1663.0088 | 27529448.3632 | annual_income_band | resolved_test | 12m | 0.1067 |
| fair | 123292 | 1702880850.0000 | 0.0902 | 0.9003 | 13811.7708 | 1207.5788 | 148884809.4051 | fico_bucket | resolved_test | 12m | 0.5768 |
| good | 67427 | 1030603600.0000 | 0.0779 | 0.9018 | 15284.7316 | 1139.3861 | 76825383.9153 | fico_bucket | resolved_test | 12m | 0.2976 |
| very_good | 34892 | 525684375.0000 | 0.0632 | 0.9027 | 15066.0431 | 928.7325 | 32405334.6831 | fico_bucket | resolved_test | 12m | 0.1255 |
| C | 69668 | 1021352375.0000 | 0.0717 | 0.9015 | 14660.2798 | 989.5361 | 68939004.3661 | grade | resolved_test | 12m | 0.2671 |
| D | 34186 | 530616225.0000 | 0.1106 | 0.9003 | 15521.4481 | 1549.4109 | 52968162.7285 | grade | resolved_test | 12m | 0.2052 |
| B | 62229 | 831078825.0000 | 0.0626 | 0.9020 | 13355.1692 | 789.0411 | 49101240.6378 | grade | resolved_test | 12m | 0.1902 |
| E | 13242 | 221467875.0000 | 0.1624 | 0.8992 | 16724.6545 | 2452.6883 | 32478498.4161 | grade | resolved_test | 12m | 0.1258 |
| A | 39707 | 524738475.0000 | 0.0539 | 0.9010 | 13215.2637 | 649.5210 | 25790529.0225 | grade | resolved_test | 12m | 0.0999 |
| 2017 | 169300 | 2421184400.0000 | 0.0832 | 0.9010 | 14301.1483 | 1151.6067 | 194967021.0304 | issue_year | resolved_test | 12m | 0.7553 |
| 2018 | 56311 | 837984425.0000 | 0.0797 | 0.9016 | 14881.3629 | 1121.4240 | 63148506.9731 | issue_year | resolved_test | 12m | 0.2447 |
| 2017Q1 | 46871 | 680667100.0000 | 0.0827 | 0.9007 | 14522.1374 | 1155.5494 | 54161755.9021 | issue_year_quarter | resolved_test | 12m | 0.2098 |
| 2017Q3 | 43848 | 615148275.0000 | 0.0853 | 0.9010 | 14029.1068 | 1170.4413 | 51321508.1022 | issue_year_quarter | resolved_test | 12m | 0.1988 |
| 2017Q2 | 44487 | 629368300.0000 | 0.0827 | 0.9009 | 14147.2408 | 1131.8899 | 50354384.1473 | issue_year_quarter | resolved_test | 12m | 0.1951 |
| 2017Q4 | 34094 | 496000725.0000 | 0.0820 | 0.9015 | 14548.0356 | 1147.6909 | 39129372.8788 | issue_year_quarter | resolved_test | 12m | 0.1516 |
| 2018Q1 | 22526 | 343417875.0000 | 0.0791 | 0.9017 | 15245.3998 | 1144.1345 | 25772774.4698 | issue_year_quarter | resolved_test | 12m | 0.0998 |
| debt_consolidation | 123773 | 1929492075.0000 | 0.0851 | 0.9012 | 15588.9578 | 1269.3670 | 157113360.2402 | purpose_group | resolved_test | 12m | 0.6087 |
| credit_card | 43671 | 624033700.0000 | 0.0751 | 0.9013 | 14289.4301 | 1029.5067 | 44959585.3832 | purpose_group | resolved_test | 12m | 0.1742 |
| other | 33800 | 358913250.0000 | 0.0838 | 0.9002 | 10618.7352 | 868.1863 | 29344698.5229 | purpose_group | resolved_test | 12m | 0.1137 |
| home_improvement | 18271 | 265412300.0000 | 0.0782 | 0.9016 | 14526.4244 | 1098.3842 | 20068577.4440 | purpose_group | resolved_test | 12m | 0.0778 |
| major_purchase | 6096 | 81317500.0000 | 0.0814 | 0.9012 | 13339.4849 | 1087.4846 | 6629306.4131 | purpose_group | resolved_test | 12m | 0.0257 |
| 36 | 169806 | 2080735650.0000 | 0.0719 | 0.8989 | 12253.6050 | 804.3853 | 136589453.1890 | term_months | resolved_test | 12m | 0.5292 |
| 60 | 55805 | 1178433175.0000 | 0.1140 | 0.9078 | 21116.9819 | 2177.6915 | 121526074.8144 | term_months | resolved_test | 12m | 0.4708 |
| 50-100k | 112648 | 1600802175.0000 | 0.2616 | 0.9013 | 14210.6578 | 3838.5377 | 432403592.3980 | annual_income_band | resolved_test | lifetime | 0.5144 |
| <50k | 61345 | 594303200.0000 | 0.2968 | 0.9001 | 9687.8833 | 2956.4734 | 181364861.0688 | annual_income_band | resolved_test | lifetime | 0.2158 |
| 100-150k | 35064 | 676464750.0000 | 0.2288 | 0.9020 | 19292.2870 | 4416.7966 | 154870555.6018 | annual_income_band | resolved_test | lifetime | 0.1842 |
| 150k+ | 16554 | 387598700.0000 | 0.1964 | 0.9023 | 23414.2020 | 4344.1866 | 71913665.0570 | annual_income_band | resolved_test | lifetime | 0.0856 |
| fair | 123292 | 1702880850.0000 | 0.3126 | 0.9003 | 13811.7708 | 4293.6695 | 529375097.8560 | fico_bucket | resolved_test | lifetime | 0.6298 |
| good | 67427 | 1030603600.0000 | 0.2296 | 0.9018 | 15284.7316 | 3470.9441 | 234035350.2155 | fico_bucket | resolved_test | lifetime | 0.2784 |
| very_good | 34892 | 525684375.0000 | 0.1410 | 0.9027 | 15066.0431 | 2210.8858 | 77142226.0540 | fico_bucket | resolved_test | lifetime | 0.0918 |
| C | 69668 | 1021352375.0000 | 0.2912 | 0.9015 | 14660.2798 | 4070.2942 | 283569256.5262 | grade | resolved_test | lifetime | 0.3374 |
| D | 34186 | 530616225.0000 | 0.3982 | 0.9003 | 15521.4481 | 5908.9624 | 202003788.9245 | grade | resolved_test | lifetime | 0.2403 |
| B | 62229 | 831078825.0000 | 0.1837 | 0.9020 | 13355.1692 | 2304.0737 | 143380202.1508 | grade | resolved_test | lifetime | 0.1706 |
| E | 13242 | 221467875.0000 | 0.4952 | 0.8992 | 16724.6545 | 7849.0281 | 103936830.1109 | grade | resolved_test | lifetime | 0.1237 |
| F | 4309 | 82582400.0000 | 0.5808 | 0.8987 | 19165.0963 | 10198.2545 | 43944278.6776 | grade | resolved_test | lifetime | 0.0523 |
| 2017 | 169300 | 2421184400.0000 | 0.2649 | 0.9010 | 14301.1483 | 3739.0527 | 633021628.6784 | issue_year | resolved_test | lifetime | 0.7531 |
| 2018 | 56311 | 837984425.0000 | 0.2503 | 0.9016 | 14881.3629 | 3685.4441 | 207531045.4471 | issue_year | resolved_test | lifetime | 0.2469 |
| 2017Q1 | 46871 | 680667100.0000 | 0.2649 | 0.9007 | 14522.1374 | 3742.8487 | 175431061.4695 | issue_year_quarter | resolved_test | lifetime | 0.2087 |
| 2017Q3 | 43848 | 615148275.0000 | 0.2709 | 0.9010 | 14029.1068 | 3801.2637 | 166677810.3033 | issue_year_quarter | resolved_test | lifetime | 0.1983 |
| 2017Q2 | 44487 | 629368300.0000 | 0.2642 | 0.9009 | 14147.2408 | 3677.0322 | 163580132.8086 | issue_year_quarter | resolved_test | lifetime | 0.1946 |
| 2017Q4 | 34094 | 496000725.0000 | 0.2582 | 0.9015 | 14548.0356 | 3734.7517 | 127332624.0970 | issue_year_quarter | resolved_test | lifetime | 0.1515 |
| 2018Q1 | 22526 | 343417875.0000 | 0.2505 | 0.9017 | 15245.3998 | 3758.1405 | 84655872.8017 | issue_year_quarter | resolved_test | lifetime | 0.1007 |
| debt_consolidation | 123773 | 1929492075.0000 | 0.2797 | 0.9012 | 15588.9578 | 4217.4300 | 522003968.3526 | purpose_group | resolved_test | lifetime | 0.6210 |
| credit_card | 43671 | 624033700.0000 | 0.2376 | 0.9013 | 14289.4301 | 3329.1547 | 145387515.7912 | purpose_group | resolved_test | lifetime | 0.1730 |
| other | 33800 | 358913250.0000 | 0.2484 | 0.9002 | 10618.7352 | 2745.0258 | 92781872.6560 | purpose_group | resolved_test | lifetime | 0.1104 |
| home_improvement | 18271 | 265412300.0000 | 0.2234 | 0.9016 | 14526.4244 | 3261.0022 | 59581771.2361 | purpose_group | resolved_test | lifetime | 0.0709 |
| major_purchase | 6096 | 81317500.0000 | 0.2422 | 0.9012 | 13339.4849 | 3411.6709 | 20797546.0896 | purpose_group | resolved_test | lifetime | 0.0247 |
| 60 | 55805 | 1178433175.0000 | 0.4047 | 0.9078 | 21116.9819 | 7658.5865 | 427387417.1362 | term_months | resolved_test | lifetime | 0.5085 |
| 36 | 169806 | 2080735650.0000 | 0.2141 | 0.8989 | 12253.6050 | 2433.1605 | 413165256.9893 | term_months | resolved_test | lifetime | 0.4915 |

### Top Segment Contributions: active_snapshot

| segment_value | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | segment_type | analysis_scope | pd_measure | portfolio_el_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50-100k | 446013 | 7067911725.0000 | 0.0856 | 0.9051 | 10317.2811 | 856.6162 | 382061980.3564 | annual_income_band | active_snapshot | 12m | 0.4916 |
| <50k | 249876 | 2694397000.0000 | 0.0911 | 0.9032 | 7170.8603 | 636.1946 | 158969766.9951 | annual_income_band | active_snapshot | 12m | 0.2045 |
| 100-150k | 143015 | 2994228925.0000 | 0.0806 | 0.9061 | 13720.2669 | 1051.3301 | 150355967.6472 | annual_income_band | active_snapshot | 12m | 0.1934 |
| 150k+ | 73705 | 1835367275.0000 | 0.0769 | 0.9060 | 16243.5829 | 1164.9539 | 85862930.0228 | annual_income_band | active_snapshot | 12m | 0.1105 |
| fair | 482110 | 7207720450.0000 | 0.0947 | 0.9041 | 9518.5603 | 865.8129 | 417417049.7875 | fico_bucket | active_snapshot | 12m | 0.5370 |
| good | 293792 | 5045942275.0000 | 0.0800 | 0.9055 | 11387.8637 | 870.8121 | 255837619.7532 | fico_bucket | active_snapshot | 12m | 0.3292 |
| very_good | 136707 | 2338242200.0000 | 0.0654 | 0.9060 | 11837.3473 | 760.7217 | 103995975.4807 | fico_bucket | active_snapshot | 12m | 0.1338 |
| C | 267738 | 4354352175.0000 | 0.0851 | 0.9053 | 10674.7606 | 836.3568 | 223924509.6145 | grade | active_snapshot | 12m | 0.2881 |
| B | 270462 | 4202846950.0000 | 0.0677 | 0.9058 | 9942.6817 | 653.0712 | 176630934.1983 | grade | active_snapshot | 12m | 0.2273 |
| D | 122780 | 2022465125.0000 | 0.1253 | 0.9039 | 11194.1378 | 1287.5637 | 158087075.3732 | grade | active_snapshot | 12m | 0.2034 |
| A | 197839 | 3057105525.0000 | 0.0567 | 0.9041 | 10176.1029 | 540.3517 | 106902641.5198 | grade | active_snapshot | 12m | 0.1375 |
| E | 41453 | 712857775.0000 | 0.1783 | 0.9031 | 11030.8005 | 1836.6298 | 76133814.7622 | grade | active_snapshot | 12m | 0.0980 |
| 2018 | 438931 | 7098278725.0000 | 0.0827 | 0.9028 | 13579.4969 | 1052.7991 | 462106153.6584 | issue_year | active_snapshot | 12m | 0.5945 |
| 2017 | 274279 | 4163772675.0000 | 0.0851 | 0.9021 | 9065.4904 | 762.4859 | 209133876.1207 | issue_year | active_snapshot | 12m | 0.2691 |
| 2016 | 141312 | 2160296550.0000 | 0.0863 | 0.9153 | 5358.6042 | 503.6725 | 71174973.1293 | issue_year | active_snapshot | 12m | 0.0916 |
| 2015 | 45550 | 919042025.0000 | 0.1073 | 0.9111 | 6940.9285 | 692.0972 | 31525029.1617 | issue_year | active_snapshot | 12m | 0.0406 |
| 2014 | 12527 | 250362675.0000 | 0.1122 | 0.8967 | 2598.0339 | 264.0887 | 3308239.2616 | issue_year | active_snapshot | 12m | 0.0043 |
| 2018Q4 | 123382 | 1980878050.0000 | 0.0784 | 0.9031 | 14806.2035 | 1078.2627 | 133038209.8593 | issue_year_quarter | active_snapshot | 12m | 0.1712 |
| 2018Q3 | 117886 | 1913733700.0000 | 0.0836 | 0.9028 | 14005.2633 | 1094.3002 | 129002677.0367 | issue_year_quarter | active_snapshot | 12m | 0.1660 |
| 2018Q2 | 112325 | 1805303150.0000 | 0.0849 | 0.9025 | 12894.9546 | 1038.1700 | 116612450.8035 | issue_year_quarter | active_snapshot | 12m | 0.1500 |
| 2018Q1 | 85338 | 1398363825.0000 | 0.0846 | 0.9026 | 12118.7855 | 977.9092 | 83452815.9590 | issue_year_quarter | active_snapshot | 12m | 0.1074 |
| 2017Q4 | 84554 | 1321353400.0000 | 0.0858 | 0.9025 | 10641.1332 | 885.5239 | 74874583.7988 | issue_year_quarter | active_snapshot | 12m | 0.0963 |
| debt_consolidation | 496456 | 8506278375.0000 | 0.0893 | 0.9051 | 11238.3910 | 947.7122 | 470497429.1641 | purpose_group | active_snapshot | 12m | 0.6053 |
| credit_card | 221352 | 3543144225.0000 | 0.0772 | 0.9051 | 10468.1876 | 768.6600 | 170144433.8592 | purpose_group | active_snapshot | 12m | 0.2189 |
| other | 111165 | 1284960975.0000 | 0.0886 | 0.9030 | 7576.6031 | 647.8439 | 72017562.1913 | purpose_group | active_snapshot | 12m | 0.0927 |
| home_improvement | 62739 | 966921400.0000 | 0.0809 | 0.9050 | 9952.2009 | 778.0029 | 48811123.5890 | purpose_group | active_snapshot | 12m | 0.0628 |
| major_purchase | 20897 | 290599950.0000 | 0.0848 | 0.9042 | 9079.7479 | 755.1369 | 15780096.2178 | purpose_group | active_snapshot | 12m | 0.0203 |
| 60 | 326036 | 6904183575.0000 | 0.1069 | 0.9116 | 14972.3786 | 1445.5448 | 471299651.1315 | term_months | active_snapshot | 12m | 0.6064 |
| 36 | 586573 | 7687721350.0000 | 0.0737 | 0.9011 | 7963.8343 | 521.5907 | 305950993.8899 | term_months | active_snapshot | 12m | 0.3936 |
| 50-100k | 446013 | 7067911725.0000 | 0.2625 | 0.9051 | 10317.2811 | 2773.8103 | 1237155459.1298 | annual_income_band | active_snapshot | lifetime | 0.5038 |
| <50k | 249876 | 2694397000.0000 | 0.2987 | 0.9032 | 7170.8603 | 2212.9173 | 552954916.7500 | annual_income_band | active_snapshot | lifetime | 0.2252 |
| 100-150k | 143015 | 2994228925.0000 | 0.2253 | 0.9061 | 13720.2669 | 3072.3578 | 439393246.0837 | annual_income_band | active_snapshot | lifetime | 0.1789 |
| 150k+ | 73705 | 1835367275.0000 | 0.1973 | 0.9060 | 16243.5829 | 3069.5179 | 226238814.3200 | annual_income_band | active_snapshot | lifetime | 0.0921 |
| fair | 482110 | 7207720450.0000 | 0.3148 | 0.9041 | 9518.5603 | 3005.9846 | 1449215254.7983 | fico_bucket | active_snapshot | lifetime | 0.5901 |
| good | 293792 | 5045942275.0000 | 0.2296 | 0.9055 | 11387.8637 | 2609.9638 | 766786485.8214 | fico_bucket | active_snapshot | lifetime | 0.3122 |
| very_good | 136707 | 2338242200.0000 | 0.1408 | 0.9060 | 11837.3473 | 1753.6827 | 239740695.6638 | fico_bucket | active_snapshot | lifetime | 0.0976 |
| C | 267738 | 4354352175.0000 | 0.3183 | 0.9053 | 10674.7606 | 3287.1780 | 880102453.4174 | grade | active_snapshot | lifetime | 0.3584 |
| D | 122780 | 2022465125.0000 | 0.4268 | 0.9039 | 11194.1378 | 4607.3977 | 565696290.0066 | grade | active_snapshot | lifetime | 0.2304 |
| B | 270462 | 4202846950.0000 | 0.2030 | 0.9058 | 9942.6817 | 1970.8518 | 533040512.1343 | grade | active_snapshot | lifetime | 0.2171 |
| E | 41453 | 712857775.0000 | 0.5047 | 0.9031 | 11030.8005 | 5321.1952 | 220579504.2361 | grade | active_snapshot | lifetime | 0.0898 |
| A | 197839 | 3057105525.0000 | 0.0900 | 0.9041 | 10176.1029 | 855.1456 | 169181140.5297 | grade | active_snapshot | lifetime | 0.0689 |
| 2018 | 438931 | 7098278725.0000 | 0.2474 | 0.9028 | 13579.4969 | 3284.9314 | 1441858207.4208 | issue_year | active_snapshot | lifetime | 0.5871 |
| 2017 | 274279 | 4163772675.0000 | 0.2547 | 0.9021 | 9065.4904 | 2404.4017 | 659476896.4573 | issue_year | active_snapshot | lifetime | 0.2685 |
| 2016 | 141312 | 2160296550.0000 | 0.2678 | 0.9153 | 5358.6042 | 1656.1601 | 234035298.8102 | issue_year | active_snapshot | lifetime | 0.0953 |
| 2015 | 45550 | 919042025.0000 | 0.3789 | 0.9111 | 6940.9285 | 2395.9056 | 109133499.0204 | issue_year | active_snapshot | lifetime | 0.0444 |
| 2014 | 12527 | 250362675.0000 | 0.3913 | 0.8967 | 2598.0339 | 896.4754 | 11230147.3069 | issue_year | active_snapshot | lifetime | 0.0046 |
| 2018Q4 | 123382 | 1980878050.0000 | 0.2520 | 0.9031 | 14806.2035 | 3583.6164 | 442153758.8388 | issue_year_quarter | active_snapshot | lifetime | 0.1800 |
| 2018Q3 | 117886 | 1913733700.0000 | 0.2495 | 0.9028 | 14005.2633 | 3375.7609 | 397954954.8355 | issue_year_quarter | active_snapshot | lifetime | 0.1621 |
| 2018Q2 | 112325 | 1805303150.0000 | 0.2446 | 0.9025 | 12894.9546 | 3128.5020 | 351408992.6851 | issue_year_quarter | active_snapshot | lifetime | 0.1431 |
| 2018Q1 | 85338 | 1398363825.0000 | 0.2418 | 0.9026 | 12118.7855 | 2933.5173 | 250340501.0614 | issue_year_quarter | active_snapshot | lifetime | 0.1019 |
| 2017Q4 | 84554 | 1321353400.0000 | 0.2497 | 0.9025 | 10641.1332 | 2715.2469 | 229584985.7742 | issue_year_quarter | active_snapshot | lifetime | 0.0935 |
| debt_consolidation | 496456 | 8506278375.0000 | 0.2806 | 0.9051 | 11238.3910 | 3051.3955 | 1514883597.9116 | purpose_group | active_snapshot | lifetime | 0.6169 |
| credit_card | 221352 | 3543144225.0000 | 0.2322 | 0.9051 | 10468.1876 | 2373.8659 | 525459971.8706 | purpose_group | active_snapshot | lifetime | 0.2140 |
| other | 111165 | 1284960975.0000 | 0.2545 | 0.9030 | 7576.6031 | 2006.6039 | 223064121.6672 | purpose_group | active_snapshot | lifetime | 0.0908 |
| home_improvement | 62739 | 966921400.0000 | 0.2289 | 0.9050 | 9952.2009 | 2297.3936 | 144136176.3434 | purpose_group | active_snapshot | lifetime | 0.0587 |
| major_purchase | 20897 | 290599950.0000 | 0.2444 | 0.9042 | 9079.7479 | 2306.4827 | 48198568.4907 | purpose_group | active_snapshot | lifetime | 0.0196 |
| 60 | 326036 | 6904183575.0000 | 0.3757 | 0.9116 | 14972.3786 | 5031.4505 | 1640433983.0554 | term_months | active_snapshot | lifetime | 0.6680 |
| 36 | 586573 | 7687721350.0000 | 0.1977 | 0.9011 | 7963.8343 | 1389.9522 | 815308453.2282 | term_months | active_snapshot | lifetime | 0.3320 |

## Next-Phase Bridge

These outputs are now sufficient to support the next course topics without implementing them yet.
- Risk-based pricing can consume the expected-loss estimates as the loss-premium input.
- Economic capital can use the expected-loss baseline together with a later unexpected-loss module.
- Segment-level EL concentration provides the management view needed before capital or pricing overlays are added.
