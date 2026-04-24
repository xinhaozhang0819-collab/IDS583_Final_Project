from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from loss_preprocess import LOSS_RAW_COLUMNS
from preprocess import ID_COLUMN


PANEL_PARTITION_ROW_STRIDE = 10_000_000


PANEL_OUTPUT_COLUMNS = [
    "sample_id",
    "snapshot_month",
    "issue_date",
    "loan_status",
    "last_payment_date",
    "split_label",
    "target_1m",
    "target_12m",
    "actual_default",
    "actual_default_12m",
    "loan_amnt",
    "funded_amnt",
    "annual_inc",
    "fico_range_low",
    "dti",
    "emp_length",
    "home_ownership",
    "purpose",
    "purpose_group",
    "fico_bucket",
    "annual_income_band",
    "verification_status",
    "grade",
    "term_months",
    "installment",
    "int_rate",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "month_on_book",
    "remaining_term",
    "seasoning_ratio",
    "month_bucket",
    "calendar_year",
    "calendar_quarter",
    "vintage_year",
    "vintage_quarter",
    "scheduled_balance_proxy",
    "scheduled_balance_to_funded",
    "term_month_interaction",
    "purpose_group_month_bucket",
    "fico_bucket_month_bucket",
]


def build_calendar_panel_with_dask(
    raw_csv_path,
    panel_output_path,
    counts_output_path=None,
    summary_output_path=None,
    data_cutoff="2018-12-01",
    train_end="2016-01-01",
    valid_end="2017-01-01",
    blocksize="64MB",
    sample_id_stride=PANEL_PARTITION_ROW_STRIDE,
):
    import dask.dataframe as dd

    panel_output_path = Path(panel_output_path)
    _remove_existing_path(panel_output_path)
    if counts_output_path:
        _remove_existing_path(Path(counts_output_path))

    raw = dd.read_csv(
        raw_csv_path,
        usecols=LOSS_RAW_COLUMNS,
        blocksize=blocksize,
        assume_missing=True,
        low_memory=False,
    )
    panel = raw.map_partitions(
        _raw_partition_to_calendar_panel,
        data_cutoff=data_cutoff,
        train_end=train_end,
        valid_end=valid_end,
        sample_id_stride=sample_id_stride,
        meta=_calendar_panel_meta(),
    )
    panel.to_parquet(
        panel_output_path,
        engine="pyarrow",
        write_index=False,
        overwrite=True,
    )

    written_panel = dd.read_parquet(panel_output_path, engine="pyarrow")
    split_counts = written_panel.groupby("split_label").size().compute().to_dict()
    panel_rows = int(sum(split_counts.values()))
    summary = {
        "panel_path": str(panel_output_path),
        "panel_rows": panel_rows,
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "data_cutoff": data_cutoff,
        "train_end": train_end,
        "valid_end": valid_end,
        "backend": "dask",
    }

    if counts_output_path:
        counts = (
            written_panel.groupby(["split_label", "month_bucket"])
            .agg({"sample_id": "count", "target_1m": ["sum", "mean"]})
            .compute()
            .reset_index()
        )
        counts.columns = [
            "split_label",
            "month_bucket",
            "exposure_count",
            "event_count",
            "mean_target_1m",
        ]
        counts["event_count"] = counts["event_count"].fillna(0)
        counts.to_parquet(counts_output_path, index=False)
        summary["counts_path"] = str(counts_output_path)

    if summary_output_path:
        Path(summary_output_path).write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
    return summary


def build_calendar_sample_only_with_dask(
    raw_csv_path,
    sampled_output_path,
    counts_output_path=None,
    summary_output_path=None,
    sample_summary_output_path=None,
    data_cutoff="2018-12-01",
    train_end="2016-01-01",
    valid_end="2017-01-01",
    blocksize="64MB",
    negative_to_positive_ratio=10,
    max_train_rows=750_000,
    max_validation_rows=500_000,
    max_test_rows=500_000,
    include_scoring_rows=True,
    max_scoring_rows=100_000,
    random_state=42,
    sample_id_stride=PANEL_PARTITION_ROW_STRIDE,
    candidate_negative_fraction=None,
    candidate_scoring_fraction=None,
):
    import dask
    import dask.dataframe as dd

    start_time = time.perf_counter()
    sampled_output_path = Path(sampled_output_path)
    _remove_existing_path(sampled_output_path)
    if counts_output_path:
        _remove_existing_path(Path(counts_output_path))
    candidate_output_path = sampled_output_path.parent / f"{sampled_output_path.stem}_candidate_parts"
    _remove_existing_path(candidate_output_path)
    candidate_output_path.mkdir(parents=True, exist_ok=True)

    raw = dd.read_csv(
        raw_csv_path,
        usecols=LOSS_RAW_COLUMNS,
        blocksize=blocksize,
        assume_missing=True,
        low_memory=False,
    )
    candidate_negative_fraction = (
        float(candidate_negative_fraction)
        if candidate_negative_fraction is not None
        else min(0.25, max(0.02, float(negative_to_positive_ratio) * 0.0075))
    )
    candidate_scoring_fraction = (
        float(candidate_scoring_fraction)
        if candidate_scoring_fraction is not None
        else min(1.0, max(0.05, float(max_scoring_rows) / 1_000_000.0))
    )
    delayed_partitions = raw.to_delayed()
    delayed_summaries = [
        dask.delayed(_write_calendar_partition_candidate_sample)(
            partition,
            partition_index=index,
            output_dir=str(candidate_output_path),
            data_cutoff=data_cutoff,
            train_end=train_end,
            valid_end=valid_end,
            sample_id_stride=sample_id_stride,
            random_state=random_state,
            candidate_negative_fraction=candidate_negative_fraction,
            candidate_scoring_fraction=candidate_scoring_fraction,
        )
        for index, partition in enumerate(delayed_partitions)
    ]
    partition_summaries = list(dask.compute(*delayed_summaries))
    counts_summary = _summarize_partition_counts(partition_summaries)
    build_seconds = time.perf_counter() - start_time

    if counts_output_path:
        counts_summary["counts"].to_parquet(counts_output_path, index=False)

    full_summary = {
        "panel_rows": int(counts_summary["panel_rows"]),
        "split_counts": counts_summary["split_counts"],
        "event_counts": counts_summary["event_counts"],
        "event_rate": counts_summary["event_rate"],
        "data_cutoff": data_cutoff,
        "train_end": train_end,
        "valid_end": valid_end,
        "backend": "dask_sample_only",
        "counts_path": str(counts_output_path) if counts_output_path else None,
        "build_counts_seconds": round(build_seconds, 3),
        "full_panel_written": False,
    }
    if summary_output_path:
        Path(summary_output_path).write_text(
            json.dumps(full_summary, indent=2, default=str),
            encoding="utf-8",
        )

    sample_start = time.perf_counter()
    exact_sample_summary = build_modeling_sample_from_panel_dask(
        candidate_output_path,
        sampled_output_path,
        negative_to_positive_ratio=negative_to_positive_ratio,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
        include_scoring_rows=include_scoring_rows,
        max_scoring_rows=max_scoring_rows,
        random_state=random_state,
    )
    _remove_existing_path(candidate_output_path)

    sample_seconds = time.perf_counter() - sample_start
    sample_summary = {
        "modeling_sample_path": str(sampled_output_path),
        "rows": exact_sample_summary["rows"],
        "split_counts": exact_sample_summary["split_counts"],
        "event_counts": exact_sample_summary["event_counts"],
        "negative_to_positive_ratio": int(negative_to_positive_ratio),
        "max_train_rows": int(max_train_rows),
        "max_validation_rows": int(max_validation_rows),
        "max_test_rows": int(max_test_rows),
        "max_scoring_rows": int(max_scoring_rows),
        "candidate_negative_fraction": float(candidate_negative_fraction),
        "candidate_scoring_fraction": float(candidate_scoring_fraction),
        "candidate_partition_summaries": {
            "partition_count": len(partition_summaries),
            "candidate_rows": int(sum(item["candidate_rows"] for item in partition_summaries)),
        },
        "backend": "dask_sample_only",
        "sample_seconds": round(sample_seconds, 3),
        "total_seconds": round(time.perf_counter() - start_time, 3),
    }
    if sample_summary_output_path:
        Path(sample_summary_output_path).write_text(
            json.dumps(sample_summary, indent=2, default=str),
            encoding="utf-8",
        )
    return {
        "panel_summary": full_summary,
        "sample_summary": sample_summary,
    }


def build_modeling_sample_from_panel_dask(
    panel_path,
    output_path,
    negative_to_positive_ratio=20,
    max_train_rows=2_000_000,
    max_validation_rows=1_000_000,
    max_test_rows=1_000_000,
    include_scoring_rows=True,
    max_scoring_rows=500_000,
    random_state=42,
):
    import dask.dataframe as dd

    panel = dd.read_parquet(panel_path, engine="pyarrow")
    samples = []
    split_limits = {
        "train": max_train_rows,
        "validation": max_validation_rows,
        "test": max_test_rows,
    }
    for split_name, max_rows in split_limits.items():
        split_sample = _sample_split_panel(
            panel,
            split_name=split_name,
            negative_to_positive_ratio=negative_to_positive_ratio,
            max_rows=max_rows,
            random_state=random_state + len(samples),
        )
        samples.append(split_sample)

    if include_scoring_rows:
        scoring = panel[panel["split_label"] == "scoring"]
        scoring_count = int(scoring.shape[0].compute())
        if scoring_count:
            scoring_frac = min(1.0, max_scoring_rows / scoring_count)
            scoring_sample = scoring.sample(
                frac=scoring_frac,
                random_state=random_state + 10,
            ).compute()
            if len(scoring_sample) > max_scoring_rows:
                scoring_sample = scoring_sample.sample(
                    n=max_scoring_rows,
                    random_state=random_state + 11,
                )
            samples.append(scoring_sample)

    sample_df = pd.concat(samples, ignore_index=True)
    sample_df = sample_df.sort_values(["split_label", "snapshot_month", ID_COLUMN]).reset_index(
        drop=True
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(output_path, index=False)
    return {
        "modeling_sample_path": str(output_path),
        "rows": int(len(sample_df)),
        "split_counts": {
            str(key): int(value)
            for key, value in sample_df["split_label"].value_counts(dropna=False).items()
        },
        "event_counts": {
            str(key): int(value)
            for key, value in sample_df.groupby("split_label")["target_1m"].sum().fillna(0).items()
        },
        "negative_to_positive_ratio": int(negative_to_positive_ratio),
        "max_train_rows": int(max_train_rows),
        "max_validation_rows": int(max_validation_rows),
        "max_test_rows": int(max_test_rows),
        "backend": "dask",
    }


def read_modeling_sample(path):
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=["snapshot_month", "issue_date"])
    return pd.read_parquet(path)


def _sample_split_panel(
    panel,
    split_name,
    negative_to_positive_ratio,
    max_rows,
    random_state,
):
    split_panel = panel[
        (panel["split_label"] == split_name) & panel["target_1m"].notnull()
    ]
    positives = split_panel[split_panel["target_1m"] == 1]
    negatives = split_panel[split_panel["target_1m"] == 0]
    positive_count = int(positives.shape[0].compute())
    negative_count = int(negatives.shape[0].compute())
    if positive_count == 0:
        target_negative_count = min(negative_count, int(max_rows))
        negative_frac = min(1.0, target_negative_count / max(negative_count, 1))
        sample = negatives.sample(frac=negative_frac, random_state=random_state).compute()
    else:
        target_negative_count = min(
            negative_count,
            max(0, int(max_rows) - positive_count),
            positive_count * int(negative_to_positive_ratio),
        )
        negative_frac = min(1.0, target_negative_count / max(negative_count, 1))
        positive_df = positives.compute()
        negative_df = negatives.sample(
            frac=negative_frac,
            random_state=random_state,
        ).compute()
        if len(negative_df) > target_negative_count:
            negative_df = negative_df.sample(n=target_negative_count, random_state=random_state)
        sample = pd.concat([positive_df, negative_df], ignore_index=True)
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=random_state)
    return sample


def _compute_panel_counts_summary(panel):
    split_counts = panel.groupby("split_label").size().compute().to_dict()
    event_counts = (
        panel[panel["target_1m"].notnull()]
        .groupby("split_label")["target_1m"]
        .sum()
        .fillna(0)
        .compute()
        .to_dict()
    )
    counts = (
        panel.groupby(["split_label", "month_bucket"])
        .agg({"sample_id": "count", "target_1m": ["sum", "mean"]})
        .compute()
        .reset_index()
    )
    counts.columns = [
        "split_label",
        "month_bucket",
        "exposure_count",
        "event_count",
        "mean_target_1m",
    ]
    counts["event_count"] = counts["event_count"].fillna(0)
    panel_rows = int(sum(split_counts.values()))
    total_events = float(sum(event_counts.values()))
    return {
        "counts": counts,
        "panel_rows": panel_rows,
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "event_counts": {str(key): int(value) for key, value in event_counts.items()},
        "event_rate": float(total_events / panel_rows) if panel_rows else 0.0,
    }


def _summarize_partition_counts(partition_summaries):
    count_frames = [
        pd.DataFrame(summary["counts"])
        for summary in partition_summaries
        if summary.get("counts")
    ]
    if count_frames:
        counts = pd.concat(count_frames, ignore_index=True)
        counts = (
            counts.groupby(["split_label", "month_bucket"], as_index=False)
            .agg(
                exposure_count=("exposure_count", "sum"),
                event_count=("event_count", "sum"),
            )
            .reset_index(drop=True)
        )
        counts["mean_target_1m"] = np.where(
            counts["exposure_count"] > 0,
            counts["event_count"] / counts["exposure_count"],
            np.nan,
        )
    else:
        counts = pd.DataFrame(
            columns=[
                "split_label",
                "month_bucket",
                "exposure_count",
                "event_count",
                "mean_target_1m",
            ]
        )
    split_counts = (
        counts.groupby("split_label")["exposure_count"].sum().astype(int).to_dict()
        if not counts.empty
        else {}
    )
    event_counts = (
        counts.groupby("split_label")["event_count"].sum().astype(int).to_dict()
        if not counts.empty
        else {}
    )
    panel_rows = int(counts["exposure_count"].sum()) if not counts.empty else 0
    total_events = float(counts["event_count"].sum()) if not counts.empty else 0.0
    return {
        "counts": counts,
        "panel_rows": panel_rows,
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "event_counts": {str(key): int(value) for key, value in event_counts.items()},
        "event_rate": float(total_events / panel_rows) if panel_rows else 0.0,
    }


def _build_sample_plan_from_counts(
    split_counts,
    event_counts,
    negative_to_positive_ratio,
    split_max_rows,
    include_scoring_rows,
    max_scoring_rows,
):
    plan = {}
    for split_name, max_rows in split_max_rows.items():
        exposure_count = int(split_counts.get(split_name, 0))
        positive_count = int(event_counts.get(split_name, 0))
        negative_count = max(0, exposure_count - positive_count)
        target_negative_count = min(
            negative_count,
            max(0, int(max_rows) - positive_count),
            positive_count * int(negative_to_positive_ratio)
            if positive_count > 0
            else int(max_rows),
        )
        plan[split_name] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "target_negative_count": int(target_negative_count),
            "negative_fraction": float(target_negative_count / negative_count)
            if negative_count
            else 0.0,
            "max_rows": int(max_rows),
        }

    scoring_count = int(split_counts.get("scoring", 0))
    scoring_target_count = min(scoring_count, int(max_scoring_rows)) if include_scoring_rows else 0
    plan["scoring"] = {
        "positive_count": 0,
        "negative_count": scoring_count,
        "target_negative_count": int(scoring_target_count),
        "negative_fraction": float(scoring_target_count / scoring_count)
        if scoring_count
        else 0.0,
        "max_rows": int(max_scoring_rows),
    }
    return plan


def _write_calendar_partition_candidate_sample(
    partition,
    partition_index,
    output_dir,
    data_cutoff,
    train_end,
    valid_end,
    sample_id_stride,
    random_state,
    candidate_negative_fraction,
    candidate_scoring_fraction,
):
    panel = _raw_partition_to_calendar_panel(
        partition,
        data_cutoff=data_cutoff,
        train_end=train_end,
        valid_end=valid_end,
        sample_id_stride=sample_id_stride,
        partition_info={"number": partition_index},
    )
    if panel.empty:
        return {
            "partition_index": int(partition_index),
            "rows": 0,
            "candidate_rows": 0,
            "counts": [],
        }

    counts = (
        panel.groupby(["split_label", "month_bucket"], dropna=False)
        .agg(
            exposure_count=(ID_COLUMN, "size"),
            event_count=("target_1m", "sum"),
        )
        .reset_index()
    )
    counts["event_count"] = counts["event_count"].fillna(0)

    positive_mask = (
        panel["split_label"].isin(["train", "validation", "test"])
        & panel["target_1m"].fillna(0).astype(int).eq(1)
    )
    negative_mask = (
        panel["split_label"].isin(["train", "validation", "test"])
        & panel["target_1m"].fillna(0).astype(int).eq(0)
        & _stable_hash_fraction(panel, random_state + int(partition_index)).lt(
            float(candidate_negative_fraction)
        )
    )
    scoring_mask = (
        panel["split_label"].eq("scoring")
        & _stable_hash_fraction(panel, random_state + 97 + int(partition_index)).lt(
            float(candidate_scoring_fraction)
        )
    )
    candidate = _align_panel_columns(
        panel.loc[positive_mask | negative_mask | scoring_mask].copy()
    )
    if not candidate.empty:
        candidate.to_parquet(
            Path(output_dir) / f"part-{int(partition_index):05d}.parquet",
            index=False,
        )

    return {
        "partition_index": int(partition_index),
        "rows": int(len(panel)),
        "candidate_rows": int(len(candidate)),
        "counts": counts.to_dict(orient="records"),
    }


def _raw_partition_to_calendar_sample(
    partition,
    data_cutoff,
    train_end,
    valid_end,
    sample_id_stride,
    sample_plan,
    random_state,
    partition_info=None,
):
    panel = _raw_partition_to_calendar_panel(
        partition,
        data_cutoff=data_cutoff,
        train_end=train_end,
        valid_end=valid_end,
        sample_id_stride=sample_id_stride,
        partition_info=partition_info,
    )
    if panel.empty:
        return _calendar_panel_meta()

    selected_masks = []
    for split_name, plan in sample_plan.items():
        split_mask = panel["split_label"].eq(split_name)
        if split_name == "scoring":
            if plan["negative_fraction"] <= 0:
                continue
            selected_masks.append(
                split_mask
                & _stable_hash_fraction(panel, random_state + 97).lt(
                    plan["negative_fraction"]
                )
            )
            continue

        positive_mask = split_mask & panel["target_1m"].fillna(0).astype(int).eq(1)
        selected_masks.append(positive_mask)
        if plan["negative_fraction"] > 0:
            negative_mask = split_mask & panel["target_1m"].fillna(0).astype(int).eq(0)
            selected_masks.append(
                negative_mask
                & _stable_hash_fraction(panel, random_state + len(split_name)).lt(
                    plan["negative_fraction"]
                )
            )

    if not selected_masks:
        return _calendar_panel_meta()
    selected = selected_masks[0]
    for mask in selected_masks[1:]:
        selected = selected | mask
    return _align_panel_columns(panel.loc[selected].copy())


def _stable_hash_fraction(df, seed):
    hashed = pd.util.hash_pandas_object(
        df[[ID_COLUMN, "snapshot_month", "split_label"]].astype(str),
        index=False,
    ).astype("uint64")
    return ((hashed ^ np.uint64(seed)) % np.uint64(1_000_000)).astype(float) / 1_000_000.0


def _raw_partition_to_calendar_panel(
    partition,
    data_cutoff,
    train_end,
    valid_end,
    sample_id_stride,
    partition_info=None,
):
    from experiment import (
        _build_calendar_survival_panel_frame,
        prepare_pd_hazard_loan_frame,
    )

    if partition.empty:
        return _calendar_panel_meta()
    partition = partition.copy()
    partition_number = 0
    if partition_info is not None:
        partition_number = int(partition_info.get("number", 0))
    partition[ID_COLUMN] = (
        partition_number * int(sample_id_stride) + np.arange(len(partition), dtype=np.int64)
    )
    loans = prepare_pd_hazard_loan_frame(partition, require_outcomes=True)
    panel = _build_calendar_survival_panel_frame(
        loans,
        data_cutoff=data_cutoff,
        train_end=train_end,
        valid_end=valid_end,
        include_scoring_snapshot=True,
    )
    return _align_panel_columns(panel)


def _align_panel_columns(panel):
    if panel.empty:
        return _calendar_panel_meta()
    aligned = panel.copy()
    for column in PANEL_OUTPUT_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned[PANEL_OUTPUT_COLUMNS].copy()


def _calendar_panel_meta():
    dtypes = {
        "sample_id": "int64",
        "snapshot_month": "datetime64[ns]",
        "issue_date": "datetime64[ns]",
        "loan_status": "object",
        "last_payment_date": "datetime64[ns]",
        "split_label": "object",
        "target_1m": "float64",
        "target_12m": "float64",
        "actual_default": "int64",
        "actual_default_12m": "int64",
    }
    object_columns = {
        "home_ownership",
        "purpose",
        "purpose_group",
        "fico_bucket",
        "annual_income_band",
        "verification_status",
        "grade",
        "month_bucket",
        "calendar_quarter",
        "vintage_quarter",
        "term_month_interaction",
        "purpose_group_month_bucket",
        "fico_bucket_month_bucket",
    }
    meta = {}
    for column in PANEL_OUTPUT_COLUMNS:
        if column in dtypes:
            meta[column] = dtypes[column]
        elif column in object_columns:
            meta[column] = "object"
        else:
            meta[column] = "float64"
    return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in meta.items()})


def _remove_existing_path(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
