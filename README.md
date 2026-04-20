# IDS583_Final_Project

This repository is organized as a notebook-first credit-risk workflow aligned with the IDS 583 course project.

## Notebook Order

1. `notebooks/exploratory_analysis.ipynb`
   Builds the cleaned modeling dataset and documents the cleaning and feature-engineering logic.
2. `notebooks/model_comparison.ipynb`
   Runs the temporal PD model comparison, grid search, feature search, and champion-PD export.
3. `notebooks/loss_reserve_segmentation.ipynb`
   Extends the project to LGD, EAD, expected loss, CECL-style reserve analytics, and management segmentation.

## Key Outputs

- `data/processed/loan_clean.csv`
- `data/processed/test_with_pd_best_model.csv`
- `data/processed/loss_workflow_dataset.csv`
- `data/processed/charged_off_loss_proxy.csv`
- `data/processed/test_with_loss_metrics.csv`
- `data/processed/cecl_active_snapshot.csv`
- `data/processed/portfolio_expected_loss_summary.csv`
- `data/processed/segment_expected_loss_summary.csv`
- `reports/temporal_model_report.md`
- `reports/loss_reserve_segmentation_report.md`
