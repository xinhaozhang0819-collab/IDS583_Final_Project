import unittest
import tempfile

import numpy as np
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment import (  # noqa: E402
    COMPACT_CALENDAR_HAZARD_PROFILE,
    COMPACT_HAZARD_PROFILE,
    FULL_CALENDAR_HAZARD_PROFILE,
    FULL_HAZARD_PROFILE,
    PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
    TARGET_1M_COLUMN,
    _build_calendar_hazard_panel_payload,
    _build_calendar_survival_panel_frame,
    _build_hazard_panel_frame,
    _build_hazard_panel_payload,
    _aggregate_calendar_hazard_probabilities,
    _parameter_candidates,
    prepare_pd_hazard_loan_frame,
    run_experiment,
)
from calendar_bigdata import (  # noqa: E402
    build_calendar_panel_with_dask,
    build_calendar_sample_only_with_dask,
    build_modeling_sample_from_panel_dask,
    read_modeling_sample,
)
from loss_preprocess import LOSS_RAW_COLUMNS  # noqa: E402
from loss_workflow import _month_diff_to_timestamp  # noqa: E402


def _sample_hazard_raw_frame():
    statuses = [
        "Fully Paid",
        "Charged Off",
        "Current",
        "Fully Paid",
        "Charged Off",
        "Current",
        "Fully Paid",
        "Charged Off",
        "Current",
        "Fully Paid",
        "Charged Off",
        "Current",
    ]
    issues = [
        "Jan-2015",
        "Feb-2015",
        "Mar-2015",
        "Apr-2015",
        "Jan-2016",
        "Feb-2016",
        "Mar-2016",
        "Apr-2016",
        "Jan-2017",
        "Feb-2017",
        "Mar-2017",
        "Apr-2017",
    ]
    last_payments = [
        "Jan-2018",
        "Jan-2016",
        "Mar-2016",
        "Apr-2018",
        "Oct-2016",
        "Feb-2017",
        "Mar-2019",
        "Apr-2017",
        "Jan-2018",
        "Feb-2020",
        "Dec-2017",
        "Apr-2018",
    ]
    rows = []
    for index, (status, issue, last_payment) in enumerate(
        zip(statuses, issues, last_payments)
    ):
        rows.append(
            {
                "sample_id": index,
                "loan_status": status,
                "issue_d": issue,
                "last_pymnt_d": last_payment,
                "loan_amnt": 10000 + index * 100,
                "annual_inc": 50000 + index * 1000,
                "fico_range_low": 660 + index * 5,
                "dti": 10 + index,
                "emp_length": "5 years",
                "home_ownership": "RENT" if index % 2 else "MORTGAGE",
                "term": "36 months" if index % 3 else "60 months",
                "purpose": "debt_consolidation" if index % 2 else "credit_card",
                "int_rate": "10.5%",
                "installment": 300,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "open_acc": 5,
                "pub_rec": 0,
                "revol_bal": 5000,
                "revol_util": "45%",
                "total_acc": 10,
                "mort_acc": 1,
                "pub_rec_bankruptcies": 0,
                "verification_status": "Verified",
            }
        )
    return pd.DataFrame(rows)


class PDHazardExperimentTests(unittest.TestCase):
    def test_hazard_panel_marks_only_event_month_positive(self):
        loans = prepare_pd_hazard_loan_frame(
            _sample_hazard_raw_frame(),
            train_end="2016-01-01",
            valid_end="2017-01-01",
        )
        panel = _build_hazard_panel_frame(loans)

        charged_off = loans[loans["loan_status"] == "Charged Off"]
        for loan in charged_off.itertuples(index=False):
            loan_panel = panel[panel["sample_id"] == loan.sample_id]
            self.assertEqual(int(loan_panel["target"].sum()), 1)
            event_month = int(loan_panel.loc[loan_panel["target"] == 1, "month_on_book"].iat[0])
            self.assertEqual(event_month, int(loan.observed_months))

        non_default_panel = panel[panel["loan_status"] != "Charged Off"]
        self.assertEqual(int(non_default_panel["target"].sum()), 0)
        self.assertTrue((loans["months_on_book"] <= loans["term_months"]).all())

    def test_feature_profiles_keep_logistic_compact(self):
        loans = prepare_pd_hazard_loan_frame(
            _sample_hazard_raw_frame(),
            train_end="2016-01-01",
            valid_end="2017-01-01",
        )
        train = loans[loans["split_label"] == "train"]
        compact = _build_hazard_panel_payload(train, COMPACT_HAZARD_PROFILE, ["sample_id"])
        full = _build_hazard_panel_payload(train, FULL_HAZARD_PROFILE, ["sample_id"])

        self.assertLess(len(compact["feature_columns"]), len(full["feature_columns"]))
        self.assertFalse(
            any(column.startswith("purpose_") for column in compact["feature_columns"])
        )
        self.assertFalse(
            any(
                column.startswith("term_month_interaction_")
                for column in compact["feature_columns"]
            )
        )

    def test_reduced_grid_candidate_counts(self):
        config = {
            "use_grid_search": True,
            "enabled_models": [
                "logistic_regression",
                "random_forest",
                "xgboost",
                "mlp_neural_network",
            ],
            "grid_search_models": [
                "logistic_regression",
                "random_forest",
                "xgboost",
                "mlp_neural_network",
            ],
            "model_params": {
                "xgboost": {
                    "n_estimators": 120,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                }
            },
            "param_grid": {},
        }
        expected = {
            "logistic_regression": 8,
            "random_forest": 18,
            "xgboost": 24,
            "mlp_neural_network": 12,
        }
        for model_name, count in expected.items():
            candidates, _ = _parameter_candidates(model_name, config)
            self.assertEqual(len(candidates), count)
        self.assertEqual(sum(expected.values()), 62)
        xgb_candidates, _ = _parameter_candidates("xgboost", config)
        self.assertIn(
            {"n_estimators": 160, "max_depth": 6, "learning_rate": 0.08},
            [
                {
                    "n_estimators": candidate["n_estimators"],
                    "max_depth": candidate["max_depth"],
                    "learning_rate": candidate["learning_rate"],
                }
                for candidate in xgb_candidates
            ],
        )

    def test_hazard_run_outputs_compatible_pd_columns(self):
        config = {
            "enabled_models": ["logistic_regression"],
            "grid_search_models": ["logistic_regression"],
            "use_grid_search": False,
            "sample_frac": 1.0,
            "modeling_type": PD_MODELING_TYPE_CALENDAR_TIME_HAZARD,
            "data_cutoff": "2018-12-01",
            "model_params": {
                "logistic_regression": {
                    "C": 1.0,
                    "max_iter": 200,
                    "class_weight": None,
                }
            },
        }
        result = run_experiment(config, _sample_hazard_raw_frame())
        predictions = result["champion_predictions"]

        self.assertIn("predicted_hazard_1m", predictions.columns)
        self.assertIn("predicted_pd", predictions.columns)
        self.assertIn("predicted_pd_12m", predictions.columns)
        self.assertIn("snapshot_month", predictions.columns)
        self.assertIn(TARGET_1M_COLUMN, predictions.columns)
        self.assertTrue(predictions["predicted_hazard_1m"].between(0, 1).all())
        self.assertTrue(predictions["predicted_pd"].between(0, 1).all())
        self.assertTrue(predictions["predicted_pd_12m"].between(0, 1).all())
        self.assertTrue((predictions["predicted_pd_12m"] <= predictions["predicted_pd"]).all())
        self.assertEqual(result["selected_model_key"], "logistic_regression")

    def test_calendar_hazard_pd_aggregation_is_stable_and_ordered(self):
        future = pd.DataFrame(
            {
                "_score_row_id": [0] * 12 + [1] * 6 + [2] * 3 + [3] * 2,
                "horizon_month": list(range(1, 13)) + list(range(1, 7)) + [1, 2, 3] + [1, 2],
                "hazard_prob": [0.1] * 12 + [0.1] * 6 + [0.0, 0.0, 0.0] + [1.0, 0.1],
            }
        )
        scores = _aggregate_calendar_hazard_probabilities(future, "predicted_pd")
        by_id = scores.set_index("_score_row_id")

        self.assertAlmostEqual(
            by_id.loc[0, "predicted_pd_12m"],
            1 - 0.9**12,
            places=10,
        )
        self.assertAlmostEqual(by_id.loc[1, "predicted_pd"], 1 - 0.9**6, places=10)
        self.assertEqual(by_id.loc[2, "predicted_pd"], 0.0)
        self.assertEqual(by_id.loc[3, "predicted_pd"], 1.0)
        self.assertTrue((scores["predicted_pd_12m"] <= scores["predicted_pd"]).all())

    def test_stage2_calendar_scoring_month_starts(self):
        issues = pd.Series(pd.to_datetime(["2017-01-01", "2018-06-01"]))
        months = _month_diff_to_timestamp(issues, pd.Timestamp("2018-12-01"))
        self.assertEqual(months.iloc[0], 23)
        self.assertEqual(months.iloc[1], 6)

    def test_calendar_panel_splits_by_snapshot_month(self):
        loans = prepare_pd_hazard_loan_frame(
            _sample_hazard_raw_frame(),
            require_outcomes=True,
        )
        panel = _build_calendar_survival_panel_frame(
            loans,
            data_cutoff="2018-12-01",
            train_end="2016-01-01",
            valid_end="2017-01-01",
        )

        self.assertFalse(panel.duplicated(["sample_id", "snapshot_month"]).any())
        self.assertTrue((panel[panel["split_label"] == "train"]["snapshot_month"] < pd.Timestamp("2016-01-01")).all())
        self.assertTrue((panel[panel["split_label"] == "validation"]["snapshot_month"] >= pd.Timestamp("2016-01-01")).all())
        self.assertTrue((panel[panel["split_label"] == "validation"]["snapshot_month"] < pd.Timestamp("2017-01-01")).all())
        self.assertTrue((panel[panel["split_label"] == "test"]["snapshot_month"] >= pd.Timestamp("2017-01-01")).all())
        self.assertTrue((panel[panel["split_label"] == "test"]["snapshot_month"] < pd.Timestamp("2018-12-01")).all())
        self.assertTrue((panel[panel["split_label"].isin(["train", "validation", "test"])]["snapshot_month"] < pd.Timestamp("2018-12-01")).all())
        self.assertGreater(
            panel.groupby("sample_id")["split_label"].nunique().max(),
            1,
        )
        charged_off = panel[panel["loan_status"].isin(["Charged Off", "Default"])]
        self.assertTrue((charged_off.groupby("sample_id")[TARGET_1M_COLUMN].sum() <= 1).all())

    def test_calendar_feature_profiles_and_mlp_sampling(self):
        loans = prepare_pd_hazard_loan_frame(
            _sample_hazard_raw_frame(),
            require_outcomes=True,
        )
        panel = _build_calendar_survival_panel_frame(
            loans,
            data_cutoff="2018-12-01",
            train_end="2016-01-01",
            valid_end="2017-01-01",
        )
        train = panel[panel["split_label"] == "train"]
        compact = _build_calendar_hazard_panel_payload(
            train,
            COMPACT_CALENDAR_HAZARD_PROFILE,
            ["sample_id", "snapshot_month"],
        )
        full = _build_calendar_hazard_panel_payload(
            train,
            FULL_CALENDAR_HAZARD_PROFILE,
            ["sample_id", "snapshot_month"],
        )
        mlp = _build_calendar_hazard_panel_payload(
            train,
            FULL_CALENDAR_HAZARD_PROFILE,
            ["sample_id", "snapshot_month"],
            mlp_sampling_config={
                "negative_to_positive_ratio": 2,
                "max_train_rows": 20,
                "random_state": 42,
            },
            model_name="mlp_neural_network",
        )

        self.assertLess(len(compact["feature_columns"]), len(full["feature_columns"]))
        self.assertFalse(any(column.startswith("purpose_") for column in compact["feature_columns"]))
        self.assertFalse(any("purpose_group_month_bucket" in column for column in compact["feature_columns"]))
        self.assertIsNone(mlp["sample_weight"])
        self.assertLessEqual(len(mlp["df"]), 20)

    def test_dask_bigdata_panel_and_sample_smoke(self):
        raw = _sample_hazard_raw_frame()
        for column in LOSS_RAW_COLUMNS:
            if column not in raw.columns:
                raw[column] = np.nan
        raw["funded_amnt"] = raw["loan_amnt"]
        raw["grade"] = "B"
        raw["initial_list_status"] = "w"
        raw["application_type"] = "Individual"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.csv"
            panel_path = tmpdir / "calendar_panel.parquet"
            counts_path = tmpdir / "calendar_counts.parquet"
            sample_path = tmpdir / "modeling_sample.parquet"
            raw[LOSS_RAW_COLUMNS].to_csv(raw_path, index=False)

            summary = build_calendar_panel_with_dask(
                raw_path,
                panel_path,
                counts_output_path=counts_path,
                data_cutoff="2018-12-01",
                train_end="2016-01-01",
                valid_end="2017-01-01",
                blocksize="32KB",
            )
            self.assertGreater(summary["panel_rows"], 0)
            panel = pd.read_parquet(panel_path)
            self.assertFalse(panel.duplicated(["sample_id", "snapshot_month"]).any())
            self.assertTrue(counts_path.exists())

            sample_summary = build_modeling_sample_from_panel_dask(
                panel_path,
                sample_path,
                negative_to_positive_ratio=5,
                max_train_rows=100,
                max_validation_rows=100,
                max_test_rows=100,
                max_scoring_rows=50,
                random_state=42,
            )
            sample = read_modeling_sample(sample_path)
            self.assertEqual(sample_summary["rows"], len(sample))
            self.assertIn("train", set(sample["split_label"]))
            self.assertIn(TARGET_1M_COLUMN, sample.columns)

    def test_dask_sample_only_builder_does_not_write_full_panel(self):
        raw = _sample_hazard_raw_frame()
        for column in LOSS_RAW_COLUMNS:
            if column not in raw.columns:
                raw[column] = np.nan
        raw["funded_amnt"] = raw["loan_amnt"]
        raw["grade"] = "B"
        raw["initial_list_status"] = "w"
        raw["application_type"] = "Individual"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.csv"
            sample_path = tmpdir / "sample_only.parquet"
            counts_path = tmpdir / "counts.parquet"
            summary_path = tmpdir / "summary.json"
            sample_summary_path = tmpdir / "sample_summary.json"
            raw[LOSS_RAW_COLUMNS].to_csv(raw_path, index=False)

            result = build_calendar_sample_only_with_dask(
                raw_path,
                sample_path,
                counts_output_path=counts_path,
                summary_output_path=summary_path,
                sample_summary_output_path=sample_summary_path,
                data_cutoff="2018-12-01",
                train_end="2016-01-01",
                valid_end="2017-01-01",
                blocksize="32KB",
                negative_to_positive_ratio=3,
                max_train_rows=100,
                max_validation_rows=100,
                max_test_rows=100,
                max_scoring_rows=50,
                random_state=42,
            )
            sample = read_modeling_sample(sample_path)
            counts = pd.read_parquet(counts_path)

            self.assertTrue(sample_path.exists())
            self.assertTrue(counts_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(sample_summary_path.exists())
            self.assertFalse((tmpdir / "calendar_panel.parquet").exists())
            self.assertGreater(result["panel_summary"]["panel_rows"], len(sample))
            self.assertIn("event_count", counts.columns)
            self.assertFalse(sample.duplicated(["sample_id", "snapshot_month"]).any())
            event_rows = sample[sample[TARGET_1M_COLUMN].fillna(0).astype(int) == 1]
            non_event_rows = sample[sample[TARGET_1M_COLUMN].fillna(0).astype(int) == 0]
            self.assertGreater(len(event_rows), 0)
            self.assertLessEqual(len(non_event_rows), len(event_rows) * 3 + 50)


if __name__ == "__main__":
    unittest.main()
