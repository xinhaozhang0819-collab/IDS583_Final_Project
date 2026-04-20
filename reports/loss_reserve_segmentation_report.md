# Credit Loss, CECL Proxy, And Risk Segmentation Report

## Course Formula Alignment

This phase extends the stage-2 PD workflow into a course-aligned expected-loss framework.
The core lens is `Expected Loss = PD x LGD x EAD`, with installment-loan simplifications for EAD, credibility-weighted LGD lookups, a pooled-logistic hazard model for lifetime PD, and management-facing segmentation.
Pricing and economic capital are not implemented here; the report ends with a short bridge showing how these outputs feed those later modules.

## Data Assets And Temporal Policy

- Stage-2 PD benchmark path: `/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/data/processed/test_with_pd_best_model.csv`
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

### LGD Diagnostic Summary

| rows | mae | rmse | bias | correlation | avg_expected_lgd | avg_actual_lgd |
| --- | --- | --- | --- | --- | --- | --- |
| 48015 | 0.0692 | 0.0892 | -0.0405 | 0.0031 | 0.9011177467716552 | 0.9416539027069037 |

### EAD Summary

| scope | rows | mean | median | p90 | max |
| --- | --- | --- | --- | --- | --- |
| charged_off_ead_proxy | 268537 | 11170.0973 | 9502.7300 | 22507.5920 | 40000.0000 |
| resolved_holdout_ead_reference | 225611 | 14445.9660 | 12000.0000 | 30000.0000 | 40000.0000 |
| active_snapshot_ead_current | 912609 | 10467.6862 | 8390.5100 | 22878.2800 | 40000.0000 |

### Recovery Summary

| rows | mean_recovery_rate | median_recovery_rate | p90_recovery_rate | mean_net_recoveries |
| --- | --- | --- | --- | --- |
| 268537 | 0.0914 | 0.0846 | 0.1804 | 1008.0129 |

## LGD And EAD Diagnostics

### LGD Actual vs Expected

![LGD Actual vs Expected](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/lgd_actual_vs_expected.png)

### LGD Error Distribution

![LGD Error Distribution](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/lgd_error_histogram.png)

### LGD Distribution By Grade

![LGD Distribution By Grade](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/lgd_by_grade.png)

### EAD Distribution

![EAD Distribution](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/ead_distribution.png)

### Charged-Off EAD By Grade

![Charged-Off EAD By Grade](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/ead_by_grade.png)

### Recovery Rate Distribution

![Recovery Rate Distribution](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/recovery_rate_distribution.png)

### LGD Lookup Source Usage

![LGD Lookup Source Usage](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/lgd_lookup_source.png)

## Hazard-Based Lifetime PD

### Hazard Panel Summary

| split_label | panel_cells | exposure_count | event_count |
| --- | --- | --- | --- |
| test | 3755 | 10996348 | 46877 |
| train | 4590 | 23264938 | 151644 |
| validation | 4094 | 9771937 | 67725 |

### Hazard Parameter Search

| candidate_id | params | validation_lifetime_auc | validation_lifetime_ks | validation_lifetime_brier | validation_12m_auc | validation_12m_brier | validation_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | {'C': 0.5, 'class_weight': None} | 0.6943 | 0.2797 | 0.1628 | 0.6939 | 0.0892 | 1 |
| 3 | {'C': 2.0, 'class_weight': 'balanced'} | 0.6943 | 0.2797 | 0.1674 | 0.6939 | 0.0873 | 2 |
| 2 | {'C': 1.0, 'class_weight': None} | 0.6943 | 0.2797 | 0.1628 | 0.6939 | 0.0892 | 3 |

### Validation Hazard Metrics

| split_name | pd_measure | auc | ks | brier |
| --- | --- | --- | --- | --- |
| validation | lifetime | 0.6943 | 0.2797 | 0.1628 |
| validation | 12m | 0.6939 | 0.2819 | 0.0892 |

### Test Hazard Metrics

| split_name | pd_measure | auc | ks | brier |
| --- | --- | --- | --- | --- |
| test | lifetime | 0.6830 | 0.2685 | 0.1554 |
| test | 12m | 0.6707 | 0.2548 | 0.1380 |

### Validation Calibration By Vintage

| issue_year | loan_count | actual_default_rate | predicted_lifetime_pd | actual_default_12m_rate | predicted_12m_pd | split_name |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 292578 | 0.2315 | 0.2301 | 0.1005 | 0.0576 | validation |

### Test Calibration By Vintage

| issue_year | loan_count | actual_default_rate | predicted_lifetime_pd | actual_default_12m_rate | predicted_12m_pd | split_name |
| --- | --- | --- | --- | --- | --- | --- |
| 2017 | 168700 | 0.2285 | 0.2399 | 0.1616 | 0.0643 | test |
| 2018 | 55773 | 0.1493 | 0.2219 | 0.1490 | 0.0581 | test |

## Lifetime PD And CECL Visual Diagnostics

Vintage calibration is shown as a point snapshot rather than an interpolated time-series line because only one validation vintage and two test vintages are available.

### Hazard Calibration By Vintage

![Hazard Calibration By Vintage](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/hazard_vintage_calibration.png)

### Monthly Hazard Profile

![Monthly Hazard Profile](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/monthly_hazard_profile.png)

### 12-Month PD vs Lifetime PD

![12-Month PD vs Lifetime PD](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/active_pd_compare.png)

### Remaining Term vs Lifetime PD

![Remaining Term vs Lifetime PD](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/remaining_term_vs_lifetime_pd.png)

## Expected Loss Outputs

| analysis_scope | pd_measure | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | portfolio_el_share | actual_loss_amount | actual_loss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resolved_test | 12m | 225611 | 3259168825.0000 | 0.2057 | 0.9011 | 14445.9660 | 2958.9731 | 667576883.5728 | 1.0000 | 616265196.9735968 | 0.1891 |
| resolved_test | lifetime | 225611 | 3259168825.0000 | 0.2357 | 0.9011 | 14445.9660 | 3242.0600 | 731444401.4999 | 1.0000 | 616265196.9735968 | 0.1891 |
| active_snapshot | 12m | 912609 | 14591904925.0000 | 0.2056 | 0.9048 | 10467.6862 | 2132.6112 | 1946240196.3477 | 1.0000 |  | NA |
| active_snapshot | lifetime | 912609 | 14591904925.0000 | 0.1425 | 0.9048 | 10467.6862 | 1643.9742 | 1500305687.2018 | 1.0000 |  | NA |

### Expected Loss Concentration Summary

| analysis_scope | pd_measure | top_n | top_el_share |
| --- | --- | --- | --- |
| resolved_test | lifetime | 10 | 0.0003 |
| resolved_test | lifetime | 100 | 0.0031 |
| resolved_test | lifetime | 1000 | 0.0261 |
| active_snapshot | lifetime | 10 | 0.0001 |
| active_snapshot | lifetime | 100 | 0.0013 |
| active_snapshot | lifetime | 1000 | 0.0103 |

### Fixed-Horizon PD Input For 12-Month EL

- Locked stage-2 model: `HistGradientBoosting`
- Prediction source: `locked_stage2_model`
- Rows scored with the fixed-horizon model: `1138220`
- Mean fixed-horizon PD across scored rows: `0.2056`
- Used as the 12-month EL PD input: `True`

## Expected Loss Visual Summary

Absolute EL charts are separated by scope and paired with normalized views. The direct predicted-versus-actual comparison is limited to the resolved 12-month holdout so the horizon remains matched.

### Portfolio Expected Loss By Scope (Absolute And Normalized)

![Portfolio Expected Loss By Scope (Absolute And Normalized)](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/portfolio_el_comparison.png)

### Matched-Horizon Loss Comparison

![Matched-Horizon Loss Comparison](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/predicted_vs_actual_loss.png)

### Normalized Expected Loss Components

![Normalized Expected Loss Components](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/el_component_summary.png)

### Top Active Loans By Lifetime EL

![Top Active Loans By Lifetime EL](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/top_expected_loss_loans.png)

### Expected Loss Concentration Curve

![Expected Loss Concentration Curve](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/el_concentration_curve.png)

### Holdout Loan-Level Examples

| sample_id | issue_date | loan_status | stage2_champion_pd | pd_12m_fixed_horizon | pd_12m_hazard | lifetime_pd_hazard | expected_lgd | ead_reference | expected_loss_12m | expected_loss_lifetime | actual_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1718632 | 2017-01-01 00:00:00 | Fully Paid | 0.0893 | 0.0893 | 0.0213 | 0.0789 | 0.9003 | 15000.0000 | 1206.0296 | 1065.3661 | 0.0000 |
| 1718640 | 2017-01-01 00:00:00 | Fully Paid | 0.2372 | 0.2372 | 0.0656 | 0.2278 | 0.8983 | 21000.0000 | 4474.3114 | 4297.4187 | 0.0000 |
| 1718647 | 2017-01-01 00:00:00 | Fully Paid | 0.1219 | 0.1219 | 0.0652 | 0.2264 | 0.8983 | 17000.0000 | 1861.9152 | 3457.5101 | 0.0000 |
| 1718651 | 2017-01-01 00:00:00 | Fully Paid | 0.3486 | 0.3486 | 0.1144 | 0.4945 | 0.9040 | 30000.0000 | 9452.5645 | 13411.1463 | 0.0000 |
| 1718653 | 2017-01-01 00:00:00 | Fully Paid | 0.2037 | 0.2037 | 0.0974 | 0.3230 | 0.8969 | 2000.0000 | 365.4091 | 579.4760 | 0.0000 |
| 1718655 | 2017-01-01 00:00:00 | Fully Paid | 0.2188 | 0.2188 | 0.0822 | 0.2785 | 0.8983 | 10000.0000 | 1965.1310 | 2502.1856 | 0.0000 |
| 1718665 | 2017-01-01 00:00:00 | Fully Paid | 0.2315 | 0.2315 | 0.0666 | 0.3210 | 0.9085 | 21600.0000 | 4542.1149 | 6298.9725 | 0.0000 |
| 1718669 | 2017-01-01 00:00:00 | Fully Paid | 0.1187 | 0.1187 | 0.0595 | 0.2083 | 0.8983 | 15000.0000 | 1599.9479 | 2807.3464 | 0.0000 |
| 1718670 | 2017-01-01 00:00:00 | Fully Paid | 0.0765 | 0.0765 | 0.0345 | 0.1251 | 0.9000 | 4000.0000 | 275.3576 | 450.3651 | 0.0000 |
| 1718672 | 2017-01-01 00:00:00 | Fully Paid | 0.0425 | 0.0425 | 0.0213 | 0.0789 | 0.9003 | 6000.0000 | 229.4670 | 426.1464 | 0.0000 |

## CECL Proxy And Reserve Snapshot

The reserve view combines lifetime PD from the hazard model with credibility-weighted expected LGD and current EAD.

### Active Snapshot Examples

| sample_id | issue_date | loan_status | remaining_term | stage2_champion_pd | pd_12m_fixed_horizon | pd_12m_hazard | lifetime_pd_hazard | expected_lgd | ead_current | expected_loss_12m | lifetime_expected_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1781187 | 2013-10-01 00:00:00 | Current | 0.0 | 0.3871 | 0.3871 | 0.0000 | 0.0000 | 0.8851 | 125.0500 | 42.8477 | 0.0000 |
| 1790999 | 2013-10-01 00:00:00 | Current | 0.0 | 0.3278 | 0.3278 | 0.0000 | 0.0000 | 0.8882 | 13730.7023 | 3997.3220 | 0.0000 |
| 1751337 | 2013-12-01 00:00:00 | Late (31-120 days) | 0.0 | 0.4311 | 0.4311 | 0.0000 | 0.0000 | 0.8882 | 11.1900 | 4.2844 | 0.0000 |
| 1753200 | 2013-12-01 00:00:00 | Late (31-120 days) | 1.0 | 0.4182 | 0.4182 | 0.0097 | 0.0097 | 0.8823 | 286.3600 | 105.6628 | 2.4451 |
| 1754434 | 2013-12-01 00:00:00 | Current | 0.0 | 0.2665 | 0.2665 | 0.0000 | 0.0000 | 0.8851 | 1615.5400 | 381.0488 | 0.0000 |
| 1760475 | 2013-12-01 00:00:00 | In Grace Period | 0.0 | 0.3317 | 0.3317 | 0.0000 | 0.0000 | 0.8882 | 553.5300 | 163.0786 | 0.0000 |
| 1761504 | 2013-12-01 00:00:00 | Current | 0.0 | 0.2764 | 0.2764 | 0.0000 | 0.0000 | 0.8851 | 931.3100 | 227.8012 | 0.0000 |
| 1761719 | 2013-12-01 00:00:00 | Current | 0.0 | 0.1286 | 0.1286 | 0.0000 | 0.0000 | 0.8882 | 10314.2289 | 1178.0259 | 0.0000 |
| 1761741 | 2013-12-01 00:00:00 | Current | 0.0 | 0.6342 | 0.6342 | 0.0000 | 0.0000 | 0.8777 | 507.2800 | 282.3689 | 0.0000 |
| 1763190 | 2013-12-01 00:00:00 | Late (31-120 days) | 0.0 | 0.1815 | 0.1815 | 0.0000 | 0.0000 | 0.8878 | 803.5500 | 129.4827 | 0.0000 |

## Risk Segmentation Analysis

The management-facing cuts in this phase are `grade, fico_bucket, term_months, issue_year, issue_year_quarter, purpose_group, annual_income_band`.

### Top Segment Contributions: resolved_test

| segment_value | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | segment_type | analysis_scope | pd_measure | portfolio_el_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50-100k | 112648 | 1600802175.0000 | 0.2060 | 0.9013 | 14210.6578 | 3058.5860 | 344543596.8022 | annual_income_band | resolved_test | 12m | 0.5161 |
| <50k | 61345 | 594303200.0000 | 0.2360 | 0.9001 | 9687.8833 | 2381.9629 | 146121515.4007 | annual_income_band | resolved_test | 12m | 0.2189 |
| 100-150k | 35064 | 676464750.0000 | 0.1780 | 0.9020 | 19292.2870 | 3465.8873 | 121527870.9009 | annual_income_band | resolved_test | 12m | 0.1820 |
| 150k+ | 16554 | 387598700.0000 | 0.1507 | 0.9023 | 23414.2020 | 3345.6506 | 55383900.4690 | annual_income_band | resolved_test | 12m | 0.0830 |
| fair | 123292 | 1702880850.0000 | 0.2487 | 0.9003 | 13811.7708 | 3449.2842 | 425269151.4268 | fico_bucket | resolved_test | 12m | 0.6370 |
| good | 67427 | 1030603600.0000 | 0.1783 | 0.9018 | 15284.7316 | 2716.2506 | 183148630.1544 | fico_bucket | resolved_test | 12m | 0.2743 |
| very_good | 34892 | 525684375.0000 | 0.1070 | 0.9027 | 15066.0431 | 1695.4919 | 59159101.9916 | fico_bucket | resolved_test | 12m | 0.0886 |
| C | 69668 | 1021352375.0000 | 0.2264 | 0.9015 | 14660.2798 | 3173.4508 | 221087968.0262 | grade | resolved_test | 12m | 0.3312 |
| D | 34186 | 530616225.0000 | 0.3209 | 0.9003 | 15521.4481 | 4791.3504 | 163797105.1871 | grade | resolved_test | 12m | 0.2454 |
| B | 62229 | 831078825.0000 | 0.1377 | 0.9020 | 13355.1692 | 1732.1704 | 107791233.7083 | grade | resolved_test | 12m | 0.1615 |
| E | 13242 | 221467875.0000 | 0.4147 | 0.8992 | 16724.6545 | 6619.6520 | 87657431.6580 | grade | resolved_test | 12m | 0.1313 |
| F | 4309 | 82582400.0000 | 0.4958 | 0.8987 | 19165.0963 | 8739.2399 | 37657384.5757 | grade | resolved_test | 12m | 0.0564 |
| 2017 | 169300 | 2421184400.0000 | 0.2088 | 0.9010 | 14301.1483 | 2971.7263 | 503113255.6941 | issue_year | resolved_test | 12m | 0.7536 |
| 2018 | 56311 | 837984425.0000 | 0.1965 | 0.9016 | 14881.3629 | 2920.6306 | 164463627.8787 | issue_year | resolved_test | 12m | 0.2464 |
| 2017Q1 | 46871 | 680667100.0000 | 0.2089 | 0.9007 | 14522.1374 | 2968.6516 | 139143669.1568 | issue_year_quarter | resolved_test | 12m | 0.2084 |
| 2017Q3 | 43848 | 615148275.0000 | 0.2141 | 0.9010 | 14029.1068 | 3032.9576 | 132989124.2556 | issue_year_quarter | resolved_test | 12m | 0.1992 |
| 2017Q2 | 44487 | 629368300.0000 | 0.2081 | 0.9009 | 14147.2408 | 2919.0067 | 129857851.7523 | issue_year_quarter | resolved_test | 12m | 0.1945 |
| 2017Q4 | 34094 | 496000725.0000 | 0.2029 | 0.9015 | 14548.0356 | 2965.9943 | 101122610.5295 | issue_year_quarter | resolved_test | 12m | 0.1515 |
| 2018Q1 | 22526 | 343417875.0000 | 0.1967 | 0.9017 | 15245.3998 | 2976.6805 | 67052704.9212 | issue_year_quarter | resolved_test | 12m | 0.1004 |
| debt_consolidation | 123773 | 1929492075.0000 | 0.2214 | 0.9012 | 15588.9578 | 3362.1931 | 416148729.3680 | purpose_group | resolved_test | 12m | 0.6234 |
| credit_card | 43671 | 624033700.0000 | 0.1854 | 0.9013 | 14289.4301 | 2617.1141 | 114291992.0221 | purpose_group | resolved_test | 12m | 0.1712 |
| other | 33800 | 358913250.0000 | 0.1950 | 0.9002 | 10618.7352 | 2184.4329 | 73833831.1862 | purpose_group | resolved_test | 12m | 0.1106 |
| home_improvement | 18271 | 265412300.0000 | 0.1736 | 0.9016 | 14526.4244 | 2559.7846 | 46769823.5366 | purpose_group | resolved_test | 12m | 0.0701 |
| major_purchase | 6096 | 81317500.0000 | 0.1898 | 0.9012 | 13339.4849 | 2712.0255 | 16532507.4599 | purpose_group | resolved_test | 12m | 0.0248 |
| 60 | 55805 | 1178433175.0000 | 0.3297 | 0.9078 | 21116.9819 | 6232.4480 | 347801759.5193 | term_months | resolved_test | 12m | 0.5210 |
| 36 | 169806 | 2080735650.0000 | 0.1650 | 0.8989 | 12253.6050 | 1883.1792 | 319775124.0535 | term_months | resolved_test | 12m | 0.4790 |
| 50-100k | 112648 | 1600802175.0000 | 0.2400 | 0.9013 | 14210.6578 | 3390.6479 | 381949702.8483 | annual_income_band | resolved_test | lifetime | 0.5222 |
| <50k | 61345 | 594303200.0000 | 0.2627 | 0.9001 | 9687.8833 | 2516.5934 | 154380419.7996 | annual_income_band | resolved_test | lifetime | 0.2111 |
| 100-150k | 35064 | 676464750.0000 | 0.2030 | 0.9020 | 19292.2870 | 3780.7728 | 132569018.6328 | annual_income_band | resolved_test | lifetime | 0.1812 |
| 150k+ | 16554 | 387598700.0000 | 0.1756 | 0.9023 | 23414.2020 | 3778.2566 | 62545260.2191 | annual_income_band | resolved_test | lifetime | 0.0855 |
| fair | 123292 | 1702880850.0000 | 0.2839 | 0.9003 | 13811.7708 | 3759.5646 | 463524233.7585 | fico_bucket | resolved_test | lifetime | 0.6337 |
| good | 67427 | 1030603600.0000 | 0.2031 | 0.9018 | 15284.7316 | 2984.6984 | 201249261.6540 | fico_bucket | resolved_test | lifetime | 0.2751 |
| very_good | 34892 | 525684375.0000 | 0.1283 | 0.9027 | 15066.0431 | 1910.7791 | 66670906.0874 | fico_bucket | resolved_test | lifetime | 0.0911 |
| C | 69668 | 1021352375.0000 | 0.2614 | 0.9015 | 14660.2798 | 3493.5432 | 243388166.6678 | grade | resolved_test | lifetime | 0.3328 |
| D | 34186 | 530616225.0000 | 0.3598 | 0.9003 | 15521.4481 | 5089.2668 | 173981674.3536 | grade | resolved_test | lifetime | 0.2379 |
| B | 62229 | 831078825.0000 | 0.1554 | 0.9020 | 13355.1692 | 1865.4617 | 116085815.4195 | grade | resolved_test | lifetime | 0.1587 |
| E | 13242 | 221467875.0000 | 0.4620 | 0.8992 | 16724.6545 | 7038.8896 | 93208976.3987 | grade | resolved_test | lifetime | 0.1274 |
| F | 4309 | 82582400.0000 | 0.5847 | 0.8987 | 19165.0963 | 10113.3162 | 43578279.7078 | grade | resolved_test | lifetime | 0.0596 |
| 2017 | 169300 | 2421184400.0000 | 0.2401 | 0.9010 | 14301.1483 | 3284.3858 | 556046512.1934 | issue_year | resolved_test | lifetime | 0.7602 |
| 2018 | 56311 | 837984425.0000 | 0.2223 | 0.9016 | 14881.3629 | 3114.8069 | 175397889.3065 | issue_year | resolved_test | lifetime | 0.2398 |
| 2017Q1 | 46871 | 680667100.0000 | 0.2412 | 0.9007 | 14522.1374 | 3329.5221 | 156058029.3692 | issue_year_quarter | resolved_test | lifetime | 0.2134 |
| 2017Q3 | 43848 | 615148275.0000 | 0.2458 | 0.9010 | 14029.1068 | 3332.4524 | 146121373.1840 | issue_year_quarter | resolved_test | lifetime | 0.1998 |
| 2017Q2 | 44487 | 629368300.0000 | 0.2397 | 0.9009 | 14147.2408 | 3240.2234 | 144147818.3927 | issue_year_quarter | resolved_test | lifetime | 0.1971 |
| 2017Q4 | 34094 | 496000725.0000 | 0.2319 | 0.9015 | 14548.0356 | 3218.1408 | 109719291.2475 | issue_year_quarter | resolved_test | lifetime | 0.1500 |
| 2018Q1 | 22526 | 343417875.0000 | 0.2228 | 0.9017 | 15245.3998 | 3204.3641 | 72181506.3134 | issue_year_quarter | resolved_test | lifetime | 0.0987 |
| debt_consolidation | 123773 | 1929492075.0000 | 0.2553 | 0.9012 | 15588.9578 | 3752.6065 | 464471361.2104 | purpose_group | resolved_test | lifetime | 0.6350 |
| credit_card | 43671 | 624033700.0000 | 0.2067 | 0.9013 | 14289.4301 | 2815.3334 | 122948425.0605 | purpose_group | resolved_test | lifetime | 0.1681 |
| other | 33800 | 358913250.0000 | 0.2223 | 0.9002 | 10618.7352 | 2236.4042 | 75590460.4661 | purpose_group | resolved_test | lifetime | 0.1033 |
| home_improvement | 18271 | 265412300.0000 | 0.2026 | 0.9016 | 14526.4244 | 2799.6445 | 51152304.9560 | purpose_group | resolved_test | lifetime | 0.0699 |
| major_purchase | 6096 | 81317500.0000 | 0.2183 | 0.9012 | 13339.4849 | 2834.9491 | 17281849.8070 | purpose_group | resolved_test | lifetime | 0.0236 |
| 60 | 55805 | 1178433175.0000 | 0.3506 | 0.9078 | 21116.9819 | 6584.4896 | 367447443.4913 | term_months | resolved_test | lifetime | 0.5024 |
| 36 | 169806 | 2080735650.0000 | 0.1979 | 0.8989 | 12253.6050 | 2143.6048 | 363996958.0087 | term_months | resolved_test | lifetime | 0.4976 |

### Top Segment Contributions: active_snapshot

| segment_value | loan_count | funded_amount | avg_pd | avg_lgd | avg_ead | avg_el | total_el | segment_type | analysis_scope | pd_measure | portfolio_el_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50-100k | 446013 | 7067911725.0000 | 0.2066 | 0.9051 | 10317.2811 | 2204.8853 | 983407512.6445 | annual_income_band | active_snapshot | 12m | 0.5053 |
| <50k | 249876 | 2694397000.0000 | 0.2377 | 0.9032 | 7170.8603 | 1785.2450 | 446089879.2853 | annual_income_band | active_snapshot | 12m | 0.2292 |
| 100-150k | 143015 | 2994228925.0000 | 0.1745 | 0.9061 | 13720.2669 | 2398.2378 | 342983985.3989 | annual_income_band | active_snapshot | 12m | 0.1762 |
| 150k+ | 73705 | 1835367275.0000 | 0.1510 | 0.9060 | 16243.5829 | 2357.4903 | 173758819.0190 | annual_income_band | active_snapshot | 12m | 0.0893 |
| fair | 482110 | 7207720450.0000 | 0.2504 | 0.9041 | 9518.5603 | 2414.6694 | 1164136255.9468 | fico_bucket | active_snapshot | 12m | 0.5981 |
| good | 293792 | 5045942275.0000 | 0.1779 | 0.9055 | 11387.8637 | 2038.0496 | 598762663.2426 | fico_bucket | active_snapshot | 12m | 0.3077 |
| very_good | 136707 | 2338242200.0000 | 0.1067 | 0.9060 | 11837.3473 | 1341.1257 | 183341277.1583 | fico_bucket | active_snapshot | 12m | 0.0942 |
| C | 267738 | 4354352175.0000 | 0.2498 | 0.9053 | 10674.7606 | 2592.4710 | 694102988.3976 | grade | active_snapshot | 12m | 0.3566 |
| D | 122780 | 2022465125.0000 | 0.3470 | 0.9039 | 11194.1378 | 3771.4887 | 463063379.7553 | grade | active_snapshot | 12m | 0.2379 |
| B | 270462 | 4202846950.0000 | 0.1533 | 0.9058 | 9942.6817 | 1496.1924 | 404663200.3215 | grade | active_snapshot | 12m | 0.2079 |
| E | 41453 | 712857775.0000 | 0.4229 | 0.9031 | 11030.8005 | 4493.5045 | 186269242.9228 | grade | active_snapshot | 12m | 0.0957 |
| A | 197839 | 3057105525.0000 | 0.0653 | 0.9041 | 10176.1029 | 622.7849 | 123211149.3304 | grade | active_snapshot | 12m | 0.0633 |
| 2018 | 438931 | 7098278725.0000 | 0.1938 | 0.9028 | 13579.4969 | 2594.4531 | 1138785891.8766 | issue_year | active_snapshot | 12m | 0.5851 |
| 2017 | 274279 | 4163772675.0000 | 0.1996 | 0.9021 | 9065.4904 | 1905.5710 | 522658100.1614 | issue_year | active_snapshot | 12m | 0.2685 |
| 2016 | 141312 | 2160296550.0000 | 0.2112 | 0.9153 | 5358.6042 | 1327.3419 | 187569336.8030 | issue_year | active_snapshot | 12m | 0.0964 |
| 2015 | 45550 | 919042025.0000 | 0.3063 | 0.9111 | 6940.9285 | 1934.9665 | 88137723.0904 | issue_year | active_snapshot | 12m | 0.0453 |
| 2014 | 12527 | 250362675.0000 | 0.3176 | 0.8967 | 2598.0339 | 725.0445 | 9082632.4933 | issue_year | active_snapshot | 12m | 0.0047 |
| 2018Q4 | 123382 | 1980878050.0000 | 0.1969 | 0.9031 | 14806.2035 | 2822.1396 | 348201223.1320 | issue_year_quarter | active_snapshot | 12m | 0.1789 |
| 2018Q3 | 117886 | 1913733700.0000 | 0.1959 | 0.9028 | 14005.2633 | 2670.4222 | 314805388.4773 | issue_year_quarter | active_snapshot | 12m | 0.1618 |
| 2018Q2 | 112325 | 1805303150.0000 | 0.1919 | 0.9025 | 12894.9546 | 2475.6975 | 278082725.0774 | issue_year_quarter | active_snapshot | 12m | 0.1429 |
| 2018Q1 | 85338 | 1398363825.0000 | 0.1892 | 0.9026 | 12118.7855 | 2316.6298 | 197696555.1899 | issue_year_quarter | active_snapshot | 12m | 0.1016 |
| 2017Q4 | 84554 | 1321353400.0000 | 0.1954 | 0.9025 | 10641.1332 | 2148.2124 | 181639948.1271 | issue_year_quarter | active_snapshot | 12m | 0.0933 |
| debt_consolidation | 496456 | 8506278375.0000 | 0.2221 | 0.9051 | 11238.3910 | 2429.7840 | 1206280859.8053 | purpose_group | active_snapshot | 12m | 0.6198 |
| credit_card | 221352 | 3543144225.0000 | 0.1808 | 0.9051 | 10468.1876 | 1859.9491 | 411703463.0887 | purpose_group | active_snapshot | 12m | 0.2115 |
| other | 111165 | 1284960975.0000 | 0.1996 | 0.9030 | 7576.6031 | 1593.9097 | 177186976.3558 | purpose_group | active_snapshot | 12m | 0.0910 |
| home_improvement | 62739 | 966921400.0000 | 0.1778 | 0.9050 | 9952.2009 | 1799.8469 | 112920597.5896 | purpose_group | active_snapshot | 12m | 0.0580 |
| major_purchase | 20897 | 290599950.0000 | 0.1910 | 0.9042 | 9079.7479 | 1825.5395 | 38148299.5084 | purpose_group | active_snapshot | 12m | 0.0196 |
| 60 | 326036 | 6904183575.0000 | 0.3034 | 0.9116 | 14972.3786 | 4055.6275 | 1322280570.6674 | term_months | active_snapshot | 12m | 0.6794 |
| 36 | 586573 | 7687721350.0000 | 0.1512 | 0.9011 | 7963.8343 | 1063.7374 | 623959625.6803 | term_months | active_snapshot | 12m | 0.3206 |
| 50-100k | 446013 | 7067911725.0000 | 0.1445 | 0.9051 | 10317.2811 | 1708.9035 | 762193196.9272 | annual_income_band | active_snapshot | lifetime | 0.5080 |
| <50k | 249876 | 2694397000.0000 | 0.1617 | 0.9032 | 7170.8603 | 1364.4442 | 340941869.4187 | annual_income_band | active_snapshot | lifetime | 0.2272 |
| 100-150k | 143015 | 2994228925.0000 | 0.1208 | 0.9061 | 13720.2669 | 1822.8611 | 260696487.1847 | annual_income_band | active_snapshot | lifetime | 0.1738 |
| 150k+ | 73705 | 1835367275.0000 | 0.1076 | 0.9060 | 16243.5829 | 1851.6265 | 136474133.6712 | annual_income_band | active_snapshot | lifetime | 0.0910 |
| fair | 482110 | 7207720450.0000 | 0.1682 | 0.9041 | 9518.5603 | 1815.3718 | 875208886.9144 | fico_bucket | active_snapshot | lifetime | 0.5834 |
| good | 293792 | 5045942275.0000 | 0.1265 | 0.9055 | 11387.8637 | 1593.7227 | 468222967.0257 | fico_bucket | active_snapshot | lifetime | 0.3121 |
| very_good | 136707 | 2338242200.0000 | 0.0865 | 0.9060 | 11837.3473 | 1147.5187 | 156873833.2616 | fico_bucket | active_snapshot | lifetime | 0.1046 |
| C | 267738 | 4354352175.0000 | 0.1703 | 0.9053 | 10674.7606 | 1961.2023 | 525088388.3566 | grade | active_snapshot | lifetime | 0.3500 |
| D | 122780 | 2022465125.0000 | 0.2488 | 0.9039 | 11194.1378 | 2962.7871 | 363771002.7285 | grade | active_snapshot | lifetime | 0.2425 |
| B | 270462 | 4202846950.0000 | 0.1002 | 0.9058 | 9942.6817 | 1105.4899 | 298993013.3340 | grade | active_snapshot | lifetime | 0.1993 |
| E | 41453 | 712857775.0000 | 0.2924 | 0.9031 | 11030.8005 | 3422.4373 | 141870293.3048 | grade | active_snapshot | lifetime | 0.0946 |
| A | 197839 | 3057105525.0000 | 0.0511 | 0.9041 | 10176.1029 | 541.0863 | 107047978.3719 | grade | active_snapshot | lifetime | 0.0714 |
| 2018 | 438931 | 7098278725.0000 | 0.1831 | 0.9028 | 13579.4969 | 2358.8275 | 1035362515.8021 | issue_year | active_snapshot | lifetime | 0.6901 |
| 2017 | 274279 | 4163772675.0000 | 0.1307 | 0.9021 | 9065.4904 | 1303.9009 | 357632625.2602 | issue_year | active_snapshot | lifetime | 0.2384 |
| 2016 | 141312 | 2160296550.0000 | 0.0701 | 0.9153 | 5358.6042 | 581.0918 | 82115248.7150 | issue_year | active_snapshot | lifetime | 0.0547 |
| 2015 | 45550 | 919042025.0000 | 0.0796 | 0.9111 | 6940.9285 | 533.6147 | 24306149.2470 | issue_year | active_snapshot | lifetime | 0.0162 |
| 2014 | 12527 | 250362675.0000 | 0.0259 | 0.8967 | 2598.0339 | 70.9783 | 889145.7322 | issue_year | active_snapshot | lifetime | 0.0006 |
| 2018Q4 | 123382 | 1980878050.0000 | 0.1967 | 0.9031 | 14806.2035 | 2673.7279 | 329889891.7051 | issue_year_quarter | active_snapshot | lifetime | 0.2199 |
| 2018Q3 | 117886 | 1913733700.0000 | 0.1893 | 0.9028 | 14005.2633 | 2477.1847 | 292025390.3048 | issue_year_quarter | active_snapshot | lifetime | 0.1946 |
| 2018Q2 | 112325 | 1805303150.0000 | 0.1780 | 0.9025 | 12894.9546 | 2214.7480 | 248771569.6442 | issue_year_quarter | active_snapshot | lifetime | 0.1658 |
| 2018Q1 | 85338 | 1398363825.0000 | 0.1617 | 0.9026 | 12118.7855 | 1929.6874 | 164675664.1480 | issue_year_quarter | active_snapshot | lifetime | 0.1098 |
| 2017Q4 | 84554 | 1321353400.0000 | 0.1511 | 0.9025 | 10641.1332 | 1642.6119 | 138889405.6011 | issue_year_quarter | active_snapshot | lifetime | 0.0926 |
| debt_consolidation | 496456 | 8506278375.0000 | 0.1548 | 0.9051 | 11238.3910 | 1904.7122 | 945605784.2524 | purpose_group | active_snapshot | lifetime | 0.6303 |
| credit_card | 221352 | 3543144225.0000 | 0.1226 | 0.9051 | 10468.1876 | 1400.5781 | 310020767.2340 | purpose_group | active_snapshot | lifetime | 0.2066 |
| other | 111165 | 1284960975.0000 | 0.1386 | 0.9030 | 7576.6031 | 1161.2284 | 129087959.3090 | purpose_group | active_snapshot | lifetime | 0.0860 |
| home_improvement | 62739 | 966921400.0000 | 0.1249 | 0.9050 | 9952.2009 | 1383.1202 | 86775577.0339 | purpose_group | active_snapshot | lifetime | 0.0578 |
| major_purchase | 20897 | 290599950.0000 | 0.1357 | 0.9042 | 9079.7479 | 1378.9347 | 28815599.3725 | purpose_group | active_snapshot | lifetime | 0.0192 |
| 60 | 326036 | 6904183575.0000 | 0.2000 | 0.9116 | 14972.3786 | 3021.3111 | 985056171.9354 | term_months | active_snapshot | lifetime | 0.6566 |
| 36 | 586573 | 7687721350.0000 | 0.1106 | 0.9011 | 7963.8343 | 878.4065 | 515249515.2663 | term_months | active_snapshot | lifetime | 0.3434 |

## Risk Segmentation Dashboards

### Lifetime Expected Loss By Grade

![Lifetime Expected Loss By Grade](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/segment_el_by_grade.png)

### Lifetime Expected Loss Share By Purpose Group

![Lifetime Expected Loss Share By Purpose Group](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/segment_share_by_purpose.png)

### Expected Loss Trend By Quarter

![Expected Loss Trend By Quarter](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/vintage_el_trend.png)

### Grade x Term Heatmap

![Grade x Term Heatmap](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/grade_term_heatmap.png)

### Grade x FICO Heatmap

![Grade x FICO Heatmap](/Users/minleihao/Desktop/Risk Project/IDS583_Final_Project/reports/figures/loss_reserve/grade_fico_heatmap.png)

## Next-Phase Bridge

These outputs are now sufficient to support the next course topics without implementing them yet.
- Risk-based pricing can consume the expected-loss estimates as the loss-premium input.
- Economic capital can use the expected-loss baseline together with a later unexpected-loss module.
- Segment-level EL concentration provides the management view needed before capital or pricing overlays are added.
