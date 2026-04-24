# Credit Loss, CECL Proxy, And Risk Segmentation Report

## Course Formula Alignment

This phase extends the hazard PD workflow into a course-aligned expected-loss framework.
The core lens is `Expected Loss = PD x LGD x EAD`, with installment-loan simplifications for EAD, credibility-weighted LGD lookups, a rerun original static HGB model for lifetime PD, a calendar-time hazard model for twelve-month PD, and management-facing segmentation.
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

The main expected-loss input is the dual-PD stage2 file: static HistGradientBoosting supplies lifetime PD and calibrated calendar-time XGBoost hazard supplies twelve-month PD. The auxiliary reserve hazard benchmark below is retained only for diagnostics and comparison.

- Main PD champion used for EL: `Dual PD: Static HGB lifetime + calendar hazard 12M`
- Prediction source: `precomputed_dual_pd_static_lifetime_and_12m_hazard`
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
| observable_12m_test | 12m | 443579 | 6584957075.0000 | 0.0803 | 0.9017 | 14845.0605 | 1136.6423 | 504190654.7861 | 1.0000 | 350447341.16339725 | 0.0532 |
| resolved_test | 12m | 225611 | 3259168825.0000 | 0.0826 | 0.9011 | 14445.9660 | 1146.9823 | 258771813.8889 | 1.0000 | 477277111.178897 | 0.1464 |
| resolved_test | lifetime | 225611 | 3259168825.0000 | 0.2609 | 0.9011 | 14445.9660 | 3718.7177 | 838983618.6029 | 1.0000 | 616265196.9735968 | 0.1891 |
| active_snapshot | 12m | 912609 | 14591904925.0000 | 0.0800 | 0.9048 | 10467.6862 | 802.9108 | 732743619.4710 | 1.0000 |  | NA |
| active_snapshot | lifetime | 912609 | 14591904925.0000 | 0.2611 | 0.9048 | 10467.6862 | 2686.6876 | 2451895312.7328 | 1.0000 |  | NA |

### Fixed-Horizon PD Input For 12-Month EL

- Locked dual-PD champion: `Dual PD: Static HGB lifetime + calendar hazard 12M`
- Prediction source: `precomputed_dual_pd_static_lifetime_and_12m_hazard`
- Rows scored with the fixed-horizon model: `1138220`
- Mean compatibility `predicted_pd` across scored rows, interpreted as lifetime PD: `0.2602`
- Mean 12-month PD across scored rows: `0.0804`
- Mean lifetime PD across scored rows: `0.2602`
- Used as the 12-month EL PD input: `True`
- Stage2 scoring cap per scope: `None`
- Resolved rows scored by stage2: `225611`
- Active rows scored by stage2: `912609`
- Unscored rows fall back to reserve hazard PD: `False`
- Twelve-month hazard calibration method: `none`
- Twelve-month hazard calibration intercept shift: `NA`
- Conservative 12-month overlay method: `validation_grade_term_conservative_floor`
- Conservative overlay buffer: `1.1000`
- Conservative overlay cap: `static_lifetime_pd`
- Validation portfolio floor PD before buffer: `0.0628`

### Holdout Loan-Level Examples

| sample_id | issue_date | loan_status | stage2_model_name | auxiliary_hazard_model_name | stage2_champion_pd | pd_12m_fixed_horizon | pd_12m_hazard | lifetime_pd_hazard | expected_lgd | ead_reference | expected_loss_12m | expected_loss_lifetime | actual_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1718632 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.1157 | 0.0591 | 0.0259 | 0.1157 | 0.9003 | 15000.0000 | 798.2399 | 1562.9930 | 0.0000 |
| 1718640 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.3080 | 0.0657 | 0.0677 | 0.3080 | 0.8983 | 21000.0000 | 1240.0482 | 5810.5586 | 0.0000 |
| 1718647 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.1706 | 0.0657 | 0.0653 | 0.1706 | 0.8983 | 17000.0000 | 1003.8485 | 2605.1052 | 0.0000 |
| 1718651 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.4396 | 0.1627 | 0.1223 | 0.4396 | 0.9040 | 30000.0000 | 4413.2796 | 11921.8316 | 0.0000 |
| 1718653 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.2602 | 0.1118 | 0.1030 | 0.2602 | 0.8969 | 2000.0000 | 200.6309 | 466.7133 | 0.0000 |
| 1718655 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.2918 | 0.0657 | 0.0798 | 0.2918 | 0.8983 | 10000.0000 | 590.4991 | 2621.5508 | 0.0000 |
| 1718665 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.2997 | 0.0874 | 0.0588 | 0.2997 | 0.9085 | 21600.0000 | 1714.4704 | 5882.0686 | 0.0000 |
| 1718669 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.1448 | 0.0657 | 0.0658 | 0.1448 | 0.8983 | 15000.0000 | 885.7487 | 1950.5281 | 0.0000 |
| 1718670 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.1196 | 0.0591 | 0.0353 | 0.1196 | 0.9000 | 4000.0000 | 212.8037 | 430.4657 | 0.0000 |
| 1718672 | 2017-01-01 00:00:00 | Fully Paid | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.0601 | 0.0591 | 0.0259 | 0.0601 | 0.9003 | 6000.0000 | 319.2960 | 324.9025 | 0.0000 |

## CECL Proxy And Reserve Snapshot

The reserve view combines lifetime PD from the main stage2 champion (Dual PD: Static HGB lifetime + calendar hazard 12M) with credibility-weighted expected LGD and current EAD.

### Active Snapshot Examples

| sample_id | issue_date | loan_status | remaining_term | stage2_model_name | auxiliary_hazard_model_name | stage2_champion_pd | pd_12m_fixed_horizon | pd_12m_hazard | lifetime_pd_hazard | expected_lgd | ead_current | expected_loss_12m | lifetime_expected_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1781187 | 2013-10-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.4571 | 0.0000 | 0.0000 | 0.4571 | 0.8851 | 125.0500 | 0.0000 | 50.5941 |
| 1790999 | 2013-10-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.4004 | 0.0000 | 0.0000 | 0.4004 | 0.8882 | 13730.7023 | 0.0000 | 4883.0440 |
| 1751337 | 2013-12-01 00:00:00 | Late (31-120 days) | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.5169 | 0.0000 | 0.0000 | 0.5169 | 0.8882 | 11.1900 | 0.0000 | 5.1376 |
| 1753200 | 2013-12-01 00:00:00 | Late (31-120 days) | 1.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.4896 | 0.2284 | 0.0096 | 0.4896 | 0.8823 | 286.3600 | 57.6932 | 123.6883 |
| 1754434 | 2013-12-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.3523 | 0.0000 | 0.0000 | 0.3523 | 0.8851 | 1615.5400 | 0.0000 | 503.7085 |
| 1760475 | 2013-12-01 00:00:00 | In Grace Period | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.4036 | 0.0000 | 0.0000 | 0.4036 | 0.8882 | 553.5300 | 0.0000 | 198.4240 |
| 1761504 | 2013-12-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.3755 | 0.0000 | 0.0000 | 0.3755 | 0.8851 | 931.3100 | 0.0000 | 309.5578 |
| 1761719 | 2013-12-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.1895 | 0.0000 | 0.0000 | 0.1895 | 0.8882 | 10314.2289 | 0.0000 | 1735.7483 |
| 1761741 | 2013-12-01 00:00:00 | Current | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.7017 | 0.0000 | 0.0000 | 0.7017 | 0.8777 | 507.2800 | 0.0000 | 312.4093 |
| 1763190 | 2013-12-01 00:00:00 | Late (31-120 days) | 0.0 | Dual PD: Static HGB lifetime + calendar hazard 12M | HistGradientBoosting Hazard | 0.2507 | 0.0000 | 0.0000 | 0.2507 | 0.8878 | 803.5500 | 0.0000 | 178.8294 |

## Risk Segmentation Analysis

The management-facing cuts in this phase are `grade, fico_bucket, term_months, issue_year, issue_year_quarter, purpose_group, annual_income_band`.

### Top Segment Contributions: observable_12m_test

| segment_value | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | segment_type | analysis_scope | pd_measure | portfolio_el_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50-100k | 217236 | 3162039350.0000 | 0.0806 | 0.9019 | 14555.7797 | 1139.8764 | 247622192.1785 | annual_income_band | observable_12m_test | 12m | 0.4911 |
| 100-150k | 68800 | 1368726450.0000 | 0.0787 | 0.9028 | 19894.2798 | 1485.9839 | 102235690.2181 | annual_income_band | observable_12m_test | 12m | 0.2028 |
| <50k | 122592 | 1201104450.0000 | 0.0820 | 0.9004 | 9797.5761 | 777.1599 | 95273589.2889 | annual_income_band | observable_12m_test | 12m | 0.1890 |
| 150k+ | 34951 | 853086825.0000 | 0.0754 | 0.9030 | 24408.0806 | 1689.7709 | 59059183.1006 | annual_income_band | observable_12m_test | 12m | 0.1171 |
| fair | 246250 | 3438891625.0000 | 0.0867 | 0.9009 | 13965.0421 | 1165.8704 | 287095581.1989 | fico_bucket | observable_12m_test | 12m | 0.5694 |
| good | 135709 | 2176951125.0000 | 0.0762 | 0.9024 | 16041.3173 | 1161.4636 | 157621063.8002 | fico_bucket | observable_12m_test | 12m | 0.3126 |
| very_good | 61620 | 969114325.0000 | 0.0634 | 0.9033 | 15727.2691 | 965.1738 | 59474009.7869 | fico_bucket | observable_12m_test | 12m | 0.1180 |
| C | 145144 | 2224923850.0000 | 0.0741 | 0.9023 | 15329.0791 | 1070.0047 | 155304763.8393 | grade | observable_12m_test | 12m | 0.3080 |
| B | 133127 | 1839662450.0000 | 0.0642 | 0.9024 | 13818.8531 | 842.8878 | 112211124.6965 | grade | observable_12m_test | 12m | 0.2226 |
| D | 56646 | 891886825.0000 | 0.1116 | 0.9007 | 15744.9215 | 1584.7886 | 89771936.6355 | grade | observable_12m_test | 12m | 0.1781 |
| A | 78796 | 1096915000.0000 | 0.0548 | 0.9012 | 13920.9478 | 695.1371 | 54774020.6892 | grade | observable_12m_test | 12m | 0.1086 |
| E | 20167 | 340651725.0000 | 0.1620 | 0.8994 | 16891.5419 | 2463.9586 | 49690652.2042 | grade | observable_12m_test | 12m | 0.0986 |
| 2017 | 443579 | 6584957075.0000 | 0.0803 | 0.9017 | 14845.0605 | 1136.6423 | 504190654.7861 | issue_year | observable_12m_test | 12m | 1.0000 |
| 2017Q3 | 122701 | 1791201400.0000 | 0.0812 | 0.9017 | 14598.0994 | 1137.8678 | 139617522.3740 | issue_year_quarter | observable_12m_test | 12m | 0.2769 |
| 2017Q4 | 118648 | 1817354125.0000 | 0.0793 | 0.9022 | 15317.1914 | 1148.6955 | 136290422.0202 | issue_year_quarter | observable_12m_test | 12m | 0.2703 |
| 2017Q2 | 105451 | 1538432075.0000 | 0.0800 | 0.9016 | 14589.0705 | 1119.0404 | 118003928.1991 | issue_year_quarter | observable_12m_test | 12m | 0.2340 |
| 2017Q1 | 96779 | 1437969475.0000 | 0.0806 | 0.9013 | 14858.2799 | 1139.4908 | 110278782.1928 | issue_year_quarter | observable_12m_test | 12m | 0.2187 |
| debt_consolidation | 245083 | 3926847225.0000 | 0.0830 | 0.9018 | 16022.5198 | 1262.2754 | 309362241.5019 | purpose_group | observable_12m_test | 12m | 0.6136 |
| credit_card | 91466 | 1364811275.0000 | 0.0737 | 0.9019 | 14921.5148 | 1045.0525 | 95586769.9904 | purpose_group | observable_12m_test | 12m | 0.1896 |
| other | 61197 | 642544250.0000 | 0.0815 | 0.9005 | 10499.6037 | 829.2108 | 50745213.2230 | purpose_group | observable_12m_test | 12m | 0.1006 |
| home_improvement | 34693 | 506094600.0000 | 0.0768 | 0.9021 | 14587.8016 | 1074.5254 | 37278511.3526 | purpose_group | observable_12m_test | 12m | 0.0739 |
| major_purchase | 11140 | 144659725.0000 | 0.0790 | 0.9015 | 12985.6127 | 1006.9945 | 11217918.7182 | purpose_group | observable_12m_test | 12m | 0.0222 |
| 36 | 320419 | 3994539275.0000 | 0.0698 | 0.8991 | 12466.6118 | 787.0512 | 252186163.3748 | term_months | observable_12m_test | 12m | 0.5002 |
| 60 | 123160 | 2590417800.0000 | 0.1075 | 0.9084 | 21032.9474 | 2046.1553 | 252004491.4113 | term_months | observable_12m_test | 12m | 0.4998 |

### Top Segment Contributions: resolved_test

| segment_value | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | segment_type | analysis_scope | pd_measure | portfolio_el_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50-100k | 112648 | 1600802175.0000 | 0.0830 | 0.9013 | 14210.6578 | 1154.3684 | 130037295.4212 | annual_income_band | resolved_test | 12m | 0.5025 |
| 100-150k | 35064 | 676464750.0000 | 0.0811 | 0.9020 | 19292.2870 | 1492.4889 | 52332629.8015 | annual_income_band | resolved_test | 12m | 0.2022 |
| <50k | 61345 | 594303200.0000 | 0.0843 | 0.9001 | 9687.8833 | 796.2392 | 48845291.4366 | annual_income_band | resolved_test | 12m | 0.1888 |
| 150k+ | 16554 | 387598700.0000 | 0.0773 | 0.9023 | 23414.2020 | 1664.6489 | 27556597.2297 | annual_income_band | resolved_test | 12m | 0.1065 |
| fair | 123292 | 1702880850.0000 | 0.0907 | 0.9003 | 13811.7708 | 1212.7060 | 149516947.8436 | fico_bucket | resolved_test | 12m | 0.5778 |
| good | 67427 | 1030603600.0000 | 0.0781 | 0.9018 | 15284.7316 | 1141.5691 | 76972579.0416 | fico_bucket | resolved_test | 12m | 0.2975 |
| very_good | 34892 | 525684375.0000 | 0.0630 | 0.9027 | 15066.0431 | 925.2060 | 32282287.0036 | fico_bucket | resolved_test | 12m | 0.1248 |
| C | 69668 | 1021352375.0000 | 0.0723 | 0.9015 | 14660.2798 | 996.6364 | 69433662.1852 | grade | resolved_test | 12m | 0.2683 |
| D | 34186 | 530616225.0000 | 0.1110 | 0.9003 | 15521.4481 | 1553.1147 | 53094777.6650 | grade | resolved_test | 12m | 0.2052 |
| B | 62229 | 831078825.0000 | 0.0632 | 0.9020 | 13355.1692 | 798.2595 | 49674890.4218 | grade | resolved_test | 12m | 0.1920 |
| E | 13242 | 221467875.0000 | 0.1609 | 0.8992 | 16724.6545 | 2424.1484 | 32100573.5499 | grade | resolved_test | 12m | 0.1240 |
| A | 39707 | 524738475.0000 | 0.0540 | 0.9010 | 13215.2637 | 651.1597 | 25855599.0720 | grade | resolved_test | 12m | 0.0999 |
| 2017 | 169300 | 2421184400.0000 | 0.0837 | 0.9010 | 14301.1483 | 1156.5848 | 195809804.2656 | issue_year | resolved_test | 12m | 0.7567 |
| 2018 | 56311 | 837984425.0000 | 0.0795 | 0.9016 | 14881.3629 | 1118.1121 | 62962009.6233 | issue_year | resolved_test | 12m | 0.2433 |
| 2017Q1 | 46871 | 680667100.0000 | 0.0834 | 0.9007 | 14522.1374 | 1163.3926 | 54529374.3248 | issue_year_quarter | resolved_test | 12m | 0.2107 |
| 2017Q3 | 43848 | 615148275.0000 | 0.0855 | 0.9010 | 14029.1068 | 1171.7554 | 51379128.6740 | issue_year_quarter | resolved_test | 12m | 0.1985 |
| 2017Q2 | 44487 | 629368300.0000 | 0.0833 | 0.9009 | 14147.2408 | 1139.0315 | 50672093.2463 | issue_year_quarter | resolved_test | 12m | 0.1958 |
| 2017Q4 | 34094 | 496000725.0000 | 0.0823 | 0.9015 | 14548.0356 | 1150.6191 | 39229208.0205 | issue_year_quarter | resolved_test | 12m | 0.1516 |
| 2018Q1 | 22526 | 343417875.0000 | 0.0793 | 0.9017 | 15245.3998 | 1145.7917 | 25810102.8563 | issue_year_quarter | resolved_test | 12m | 0.0997 |
| debt_consolidation | 123773 | 1929492075.0000 | 0.0856 | 0.9012 | 15588.9578 | 1275.3670 | 157855998.4018 | purpose_group | resolved_test | 12m | 0.6100 |
| credit_card | 43671 | 624033700.0000 | 0.0758 | 0.9013 | 14289.4301 | 1038.2231 | 45340239.7680 | purpose_group | resolved_test | 12m | 0.1752 |
| other | 33800 | 358913250.0000 | 0.0835 | 0.9002 | 10618.7352 | 861.1645 | 29107359.1576 | purpose_group | resolved_test | 12m | 0.1125 |
| home_improvement | 18271 | 265412300.0000 | 0.0782 | 0.9016 | 14526.4244 | 1095.0459 | 20007584.1126 | purpose_group | resolved_test | 12m | 0.0773 |
| major_purchase | 6096 | 81317500.0000 | 0.0804 | 0.9012 | 13339.4849 | 1059.8150 | 6460632.4488 | purpose_group | resolved_test | 12m | 0.0250 |
| 36 | 169806 | 2080735650.0000 | 0.0720 | 0.8989 | 12253.6050 | 802.8801 | 136333861.0530 | term_months | resolved_test | 12m | 0.5268 |
| 60 | 55805 | 1178433175.0000 | 0.1149 | 0.9078 | 21116.9819 | 2194.0319 | 122437952.8359 | term_months | resolved_test | 12m | 0.4732 |
| 50-100k | 112648 | 1600802175.0000 | 0.2612 | 0.9013 | 14210.6578 | 3832.1644 | 431685650.4822 | annual_income_band | resolved_test | lifetime | 0.5145 |
| <50k | 61345 | 594303200.0000 | 0.2963 | 0.9001 | 9687.8833 | 2952.1897 | 181102076.4839 | annual_income_band | resolved_test | lifetime | 0.2159 |
| 100-150k | 35064 | 676464750.0000 | 0.2283 | 0.9020 | 19292.2870 | 4403.6800 | 154410636.6143 | annual_income_band | resolved_test | lifetime | 0.1840 |
| 150k+ | 16554 | 387598700.0000 | 0.1963 | 0.9023 | 23414.2020 | 4336.4296 | 71785255.0225 | annual_income_band | resolved_test | lifetime | 0.0856 |
| fair | 123292 | 1702880850.0000 | 0.3119 | 0.9003 | 13811.7708 | 4281.8926 | 527923101.7071 | fico_bucket | resolved_test | lifetime | 0.6292 |
| good | 67427 | 1030603600.0000 | 0.2294 | 0.9018 | 15284.7316 | 3467.6999 | 233816600.4916 | fico_bucket | resolved_test | lifetime | 0.2787 |
| very_good | 34892 | 525684375.0000 | 0.1411 | 0.9027 | 15066.0431 | 2213.8002 | 77243916.4042 | fico_bucket | resolved_test | lifetime | 0.0921 |
| C | 69668 | 1021352375.0000 | 0.2905 | 0.9015 | 14660.2798 | 4058.0600 | 282716924.4560 | grade | resolved_test | lifetime | 0.3370 |
| D | 34186 | 530616225.0000 | 0.3975 | 0.9003 | 15521.4481 | 5899.9124 | 201694406.6566 | grade | resolved_test | lifetime | 0.2404 |
| B | 62229 | 831078825.0000 | 0.1839 | 0.9020 | 13355.1692 | 2307.0022 | 143562442.5347 | grade | resolved_test | lifetime | 0.1711 |
| E | 13242 | 221467875.0000 | 0.4931 | 0.8992 | 16724.6545 | 7824.3864 | 103610524.4252 | grade | resolved_test | lifetime | 0.1235 |
| F | 4309 | 82582400.0000 | 0.5791 | 0.8987 | 19165.0963 | 10173.9979 | 43839757.0077 | grade | resolved_test | lifetime | 0.0523 |
| 2017 | 169300 | 2421184400.0000 | 0.2645 | 0.9010 | 14301.1483 | 3731.7575 | 631786550.8063 | issue_year | resolved_test | lifetime | 0.7530 |
| 2018 | 56311 | 837984425.0000 | 0.2500 | 0.9016 | 14881.3629 | 3679.5132 | 207197067.7966 | issue_year | resolved_test | lifetime | 0.2470 |
| 2017Q1 | 46871 | 680667100.0000 | 0.2648 | 0.9007 | 14522.1374 | 3740.3637 | 175314588.9016 | issue_year_quarter | resolved_test | lifetime | 0.2090 |
| 2017Q3 | 43848 | 615148275.0000 | 0.2701 | 0.9010 | 14029.1068 | 3788.6977 | 166126816.2374 | issue_year_quarter | resolved_test | lifetime | 0.1980 |
| 2017Q2 | 44487 | 629368300.0000 | 0.2637 | 0.9009 | 14147.2408 | 3669.6026 | 163249612.1174 | issue_year_quarter | resolved_test | lifetime | 0.1946 |
| 2017Q4 | 34094 | 496000725.0000 | 0.2578 | 0.9015 | 14548.0356 | 3727.7977 | 127095533.5499 | issue_year_quarter | resolved_test | lifetime | 0.1515 |
| 2018Q1 | 22526 | 343417875.0000 | 0.2502 | 0.9017 | 15245.3998 | 3751.6104 | 84508774.7511 | issue_year_quarter | resolved_test | lifetime | 0.1007 |
| debt_consolidation | 123773 | 1929492075.0000 | 0.2792 | 0.9012 | 15588.9578 | 4209.4261 | 521013301.6801 | purpose_group | resolved_test | lifetime | 0.6210 |
| credit_card | 43671 | 624033700.0000 | 0.2373 | 0.9013 | 14289.4301 | 3321.0796 | 145034865.4081 | purpose_group | resolved_test | lifetime | 0.1729 |
| other | 33800 | 358913250.0000 | 0.2479 | 0.9002 | 10618.7352 | 2741.7540 | 92671285.0308 | purpose_group | resolved_test | lifetime | 0.1105 |
| home_improvement | 18271 | 265412300.0000 | 0.2231 | 0.9016 | 14526.4244 | 3255.7021 | 59484933.3461 | purpose_group | resolved_test | lifetime | 0.0709 |
| major_purchase | 6096 | 81317500.0000 | 0.2419 | 0.9012 | 13339.4849 | 3408.6669 | 20779233.1378 | purpose_group | resolved_test | lifetime | 0.0248 |
| 60 | 55805 | 1178433175.0000 | 0.4043 | 0.9078 | 21116.9819 | 7648.3790 | 426817788.7621 | term_months | resolved_test | lifetime | 0.5087 |
| 36 | 169806 | 2080735650.0000 | 0.2137 | 0.8989 | 12253.6050 | 2427.2748 | 412165829.8408 | term_months | resolved_test | lifetime | 0.4913 |

### Top Segment Contributions: active_snapshot

| segment_value | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | segment_type | analysis_scope | pd_measure | portfolio_el_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50-100k | 446013 | 7067911725.0000 | 0.0804 | 0.9051 | 10317.2811 | 804.9950 | 359038241.0124 | annual_income_band | active_snapshot | 12m | 0.4900 |
| 100-150k | 143015 | 2994228925.0000 | 0.0782 | 0.9061 | 13720.2669 | 1022.4554 | 146226463.6079 | annual_income_band | active_snapshot | 12m | 0.1996 |
| <50k | 249876 | 2694397000.0000 | 0.0815 | 0.9032 | 7170.8603 | 571.4632 | 142794944.6987 | annual_income_band | active_snapshot | 12m | 0.1949 |
| 150k+ | 73705 | 1835367275.0000 | 0.0756 | 0.9060 | 16243.5829 | 1148.9583 | 84683970.1521 | annual_income_band | active_snapshot | 12m | 0.1156 |
| fair | 482110 | 7207720450.0000 | 0.0862 | 0.9041 | 9518.5603 | 793.6731 | 382637727.7334 | fico_bucket | active_snapshot | 12m | 0.5222 |
| good | 293792 | 5045942275.0000 | 0.0769 | 0.9055 | 11387.8637 | 840.8992 | 247049464.2234 | fico_bucket | active_snapshot | 12m | 0.3372 |
| very_good | 136707 | 2338242200.0000 | 0.0646 | 0.9060 | 11837.3473 | 753.8489 | 103056427.5143 | fico_bucket | active_snapshot | 12m | 0.1406 |
| C | 267738 | 4354352175.0000 | 0.0755 | 0.9053 | 10674.7606 | 770.2436 | 206223491.9380 | grade | active_snapshot | 12m | 0.2814 |
| B | 270462 | 4202846950.0000 | 0.0673 | 0.9058 | 9942.6817 | 657.3054 | 177776140.0610 | grade | active_snapshot | 12m | 0.2426 |
| D | 122780 | 2022465125.0000 | 0.1113 | 0.9039 | 11194.1378 | 1131.8417 | 138967523.9103 | grade | active_snapshot | 12m | 0.1897 |
| A | 197839 | 3057105525.0000 | 0.0568 | 0.9041 | 10176.1029 | 543.3449 | 107494808.7306 | grade | active_snapshot | 12m | 0.1467 |
| E | 41453 | 712857775.0000 | 0.1617 | 0.9031 | 11030.8005 | 1621.2690 | 67206464.8193 | grade | active_snapshot | 12m | 0.0917 |
| 2018 | 438931 | 7098278725.0000 | 0.0772 | 0.9028 | 13579.4969 | 991.5147 | 435206558.1655 | issue_year | active_snapshot | 12m | 0.5939 |
| 2017 | 274279 | 4163772675.0000 | 0.0782 | 0.9021 | 9065.4904 | 708.5876 | 194350685.9137 | issue_year | active_snapshot | 12m | 0.2652 |
| 2016 | 141312 | 2160296550.0000 | 0.0800 | 0.9153 | 5358.6042 | 480.7023 | 67929007.8104 | issue_year | active_snapshot | 12m | 0.0927 |
| 2015 | 45550 | 919042025.0000 | 0.1086 | 0.9111 | 6940.9285 | 701.9107 | 31972030.7856 | issue_year | active_snapshot | 12m | 0.0436 |
| 2014 | 12527 | 250362675.0000 | 0.1111 | 0.8967 | 2598.0339 | 262.2559 | 3285279.1028 | issue_year | active_snapshot | 12m | 0.0045 |
| 2018Q4 | 123382 | 1980878050.0000 | 0.0780 | 0.9031 | 14806.2035 | 1075.8620 | 132742009.4906 | issue_year_quarter | active_snapshot | 12m | 0.1812 |
| 2018Q3 | 117886 | 1913733700.0000 | 0.0779 | 0.9028 | 14005.2633 | 1024.9013 | 120821515.2386 | issue_year_quarter | active_snapshot | 12m | 0.1649 |
| 2018Q2 | 112325 | 1805303150.0000 | 0.0765 | 0.9025 | 12894.9546 | 942.4900 | 105865192.1620 | issue_year_quarter | active_snapshot | 12m | 0.1445 |
| 2018Q1 | 85338 | 1398363825.0000 | 0.0761 | 0.9026 | 12118.7855 | 887.9730 | 75777841.2742 | issue_year_quarter | active_snapshot | 12m | 0.1034 |
| 2017Q4 | 84554 | 1321353400.0000 | 0.0781 | 0.9025 | 10641.1332 | 812.9587 | 68738911.7432 | issue_year_quarter | active_snapshot | 12m | 0.0938 |
| debt_consolidation | 496456 | 8506278375.0000 | 0.0828 | 0.9051 | 11238.3910 | 888.0794 | 440892357.6369 | purpose_group | active_snapshot | 12m | 0.6017 |
| credit_card | 221352 | 3543144225.0000 | 0.0738 | 0.9051 | 10468.1876 | 741.2143 | 164069261.1406 | purpose_group | active_snapshot | 12m | 0.2239 |
| other | 111165 | 1284960975.0000 | 0.0814 | 0.9030 | 7576.6031 | 595.9808 | 66252211.1259 | purpose_group | active_snapshot | 12m | 0.0904 |
| home_improvement | 62739 | 966921400.0000 | 0.0774 | 0.9050 | 9952.2009 | 747.1678 | 46876559.2237 | purpose_group | active_snapshot | 12m | 0.0640 |
| major_purchase | 20897 | 290599950.0000 | 0.0788 | 0.9042 | 9079.7479 | 701.2122 | 14653230.3439 | purpose_group | active_snapshot | 12m | 0.0200 |
| 60 | 326036 | 6904183575.0000 | 0.1032 | 0.9116 | 14972.3786 | 1389.6380 | 453072018.7542 | term_months | active_snapshot | 12m | 0.6183 |
| 36 | 586573 | 7687721350.0000 | 0.0671 | 0.9011 | 7963.8343 | 476.7891 | 279671600.7168 | term_months | active_snapshot | 12m | 0.3817 |
| 50-100k | 446013 | 7067911725.0000 | 0.2623 | 0.9051 | 10317.2811 | 2770.5848 | 1235716850.5670 | annual_income_band | active_snapshot | lifetime | 0.5040 |
| <50k | 249876 | 2694397000.0000 | 0.2985 | 0.9032 | 7170.8603 | 2211.8232 | 552681541.3707 | annual_income_band | active_snapshot | lifetime | 0.2254 |
| 100-150k | 143015 | 2994228925.0000 | 0.2247 | 0.9061 | 13720.2669 | 3061.1129 | 437785064.2709 | annual_income_band | active_snapshot | lifetime | 0.1785 |
| 150k+ | 73705 | 1835367275.0000 | 0.1972 | 0.9060 | 16243.5829 | 3062.3683 | 225711856.5241 | annual_income_band | active_snapshot | lifetime | 0.0921 |
| fair | 482110 | 7207720450.0000 | 0.3143 | 0.9041 | 9518.5603 | 2997.7774 | 1445258479.7095 | fico_bucket | active_snapshot | lifetime | 0.5894 |
| good | 293792 | 5045942275.0000 | 0.2295 | 0.9055 | 11387.8637 | 2607.6409 | 766104037.0495 | fico_bucket | active_snapshot | lifetime | 0.3125 |
| very_good | 136707 | 2338242200.0000 | 0.1411 | 0.9060 | 11837.3473 | 1759.4768 | 240532795.9738 | fico_bucket | active_snapshot | lifetime | 0.0981 |
| C | 267738 | 4354352175.0000 | 0.3179 | 0.9053 | 10674.7606 | 3281.2367 | 878511743.9946 | grade | active_snapshot | lifetime | 0.3583 |
| D | 122780 | 2022465125.0000 | 0.4256 | 0.9039 | 11194.1378 | 4594.8704 | 564158193.3463 | grade | active_snapshot | lifetime | 0.2301 |
| B | 270462 | 4202846950.0000 | 0.2033 | 0.9058 | 9942.6817 | 1972.4944 | 533484791.7216 | grade | active_snapshot | lifetime | 0.2176 |
| E | 41453 | 712857775.0000 | 0.5034 | 0.9031 | 11030.8005 | 5307.1468 | 219997156.0366 | grade | active_snapshot | lifetime | 0.0897 |
| A | 197839 | 3057105525.0000 | 0.0901 | 0.9041 | 10176.1029 | 853.2998 | 168815975.0293 | grade | active_snapshot | lifetime | 0.0689 |
| 2018 | 438931 | 7098278725.0000 | 0.2472 | 0.9028 | 13579.4969 | 3279.6464 | 1439538463.6472 | issue_year | active_snapshot | lifetime | 0.5871 |
| 2017 | 274279 | 4163772675.0000 | 0.2544 | 0.9021 | 9065.4904 | 2399.6168 | 658164499.6993 | issue_year | active_snapshot | lifetime | 0.2684 |
| 2016 | 141312 | 2160296550.0000 | 0.2676 | 0.9153 | 5358.6042 | 1655.0266 | 233875121.1922 | issue_year | active_snapshot | lifetime | 0.0954 |
| 2015 | 45550 | 919042025.0000 | 0.3788 | 0.9111 | 6940.9285 | 2394.9181 | 109088517.3486 | issue_year | active_snapshot | lifetime | 0.0445 |
| 2014 | 12527 | 250362675.0000 | 0.3911 | 0.8967 | 2598.0339 | 895.6981 | 11220409.7042 | issue_year | active_snapshot | lifetime | 0.0046 |
| 2018Q4 | 123382 | 1980878050.0000 | 0.2516 | 0.9031 | 14806.2035 | 3577.6572 | 441418498.1605 | issue_year_quarter | active_snapshot | lifetime | 0.1800 |
| 2018Q3 | 117886 | 1913733700.0000 | 0.2491 | 0.9028 | 14005.2633 | 3368.6569 | 397117489.6026 | issue_year_quarter | active_snapshot | lifetime | 0.1620 |
| 2018Q2 | 112325 | 1805303150.0000 | 0.2445 | 0.9025 | 12894.9546 | 3125.7017 | 351094439.3442 | issue_year_quarter | active_snapshot | lifetime | 0.1432 |
| 2018Q1 | 85338 | 1398363825.0000 | 0.2415 | 0.9026 | 12118.7855 | 2928.4497 | 249908036.5400 | issue_year_quarter | active_snapshot | lifetime | 0.1019 |
| 2017Q4 | 84554 | 1321353400.0000 | 0.2494 | 0.9025 | 10641.1332 | 2710.5737 | 229189850.8104 | issue_year_quarter | active_snapshot | lifetime | 0.0935 |
| debt_consolidation | 496456 | 8506278375.0000 | 0.2803 | 0.9051 | 11238.3910 | 3046.6262 | 1512515845.2226 | purpose_group | active_snapshot | lifetime | 0.6169 |
| credit_card | 221352 | 3543144225.0000 | 0.2321 | 0.9051 | 10468.1876 | 2368.3629 | 524241858.8643 | purpose_group | active_snapshot | lifetime | 0.2138 |
| other | 111165 | 1284960975.0000 | 0.2542 | 0.9030 | 7576.6031 | 2006.0430 | 223001773.2176 | purpose_group | active_snapshot | lifetime | 0.0910 |
| home_improvement | 62739 | 966921400.0000 | 0.2287 | 0.9050 | 9952.2009 | 2294.8254 | 143975047.9253 | purpose_group | active_snapshot | lifetime | 0.0587 |
| major_purchase | 20897 | 290599950.0000 | 0.2440 | 0.9042 | 9079.7479 | 2304.6747 | 48160787.5030 | purpose_group | active_snapshot | lifetime | 0.0196 |
| 60 | 326036 | 6904183575.0000 | 0.3754 | 0.9116 | 14972.3786 | 5024.5707 | 1638190939.1223 | term_months | active_snapshot | lifetime | 0.6681 |
| 36 | 586573 | 7687721350.0000 | 0.1975 | 0.9011 | 7963.8343 | 1387.2176 | 813704373.6105 | term_months | active_snapshot | lifetime | 0.3319 |

## Next-Phase Bridge

These outputs are now sufficient to support the next course topics without implementing them yet.
- Risk-based pricing can consume the expected-loss estimates as the loss-premium input.
- Economic capital can use the expected-loss baseline together with a later unexpected-loss module.
- Segment-level EL concentration provides the management view needed before capital or pricing overlays are added.
