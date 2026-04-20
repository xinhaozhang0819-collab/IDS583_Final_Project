import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_TEMPORAL_SPLIT = {
    "train_end": "2016-01-01",
    "valid_end": "2017-01-01",
}


def assign_temporal_split_labels(
    df,
    date_col="issue_date",
    train_end=DEFAULT_TEMPORAL_SPLIT["train_end"],
    valid_end=DEFAULT_TEMPORAL_SPLIT["valid_end"],
    output_col="split_label",
):
    labeled = df.copy()
    labeled[date_col] = pd.to_datetime(labeled[date_col], errors="coerce")

    train_cutoff = pd.Timestamp(train_end)
    valid_cutoff = pd.Timestamp(valid_end)

    labeled[output_col] = "test"
    labeled.loc[labeled[date_col] < train_cutoff, output_col] = "train"
    labeled.loc[
        (labeled[date_col] >= train_cutoff) & (labeled[date_col] < valid_cutoff),
        output_col,
    ] = "validation"
    labeled.loc[labeled[date_col].isna(), output_col] = pd.NA
    return labeled


def temporal_train_valid_test_split(
    df,
    feature_columns,
    target_col="default",
    date_col="issue_date",
    meta_columns=None,
    train_end=DEFAULT_TEMPORAL_SPLIT["train_end"],
    valid_end=DEFAULT_TEMPORAL_SPLIT["valid_end"],
    sample_frac=1.0,
    random_state=42,
):
    meta_columns = meta_columns or []
    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    train_cutoff = pd.Timestamp(train_end)
    valid_cutoff = pd.Timestamp(valid_end)

    train_df = working[working[date_col] < train_cutoff].copy()
    valid_df = working[(working[date_col] >= train_cutoff) & (working[date_col] < valid_cutoff)].copy()
    test_df = working[working[date_col] >= valid_cutoff].copy()

    train_df = _sample_partition(train_df, target_col, sample_frac, random_state)
    valid_df = _sample_partition(valid_df, target_col, sample_frac, random_state + 1)
    test_df = _sample_partition(test_df, target_col, sample_frac, random_state + 2)

    for name, partition in {
        "train": train_df,
        "validation": valid_df,
        "test": test_df,
    }.items():
        if partition.empty:
            raise ValueError(
                f"The temporal {name} partition is empty. Regenerate the processed dataset "
                "with issue_date included, or adjust the temporal boundaries."
            )

    split_summary = pd.DataFrame(
        [
            _summarize_partition("train", train_df, target_col, date_col),
            _summarize_partition("validation", valid_df, target_col, date_col),
            _summarize_partition("test", test_df, target_col, date_col),
        ]
    )

    return {
        "train": _extract_partition_payload(train_df, feature_columns, target_col, meta_columns),
        "validation": _extract_partition_payload(
            valid_df,
            feature_columns,
            target_col,
            meta_columns,
        ),
        "test": _extract_partition_payload(test_df, feature_columns, target_col, meta_columns),
        "split_summary": split_summary,
        "temporal_bounds": {
            "train_end": train_cutoff,
            "valid_end": valid_cutoff,
        },
    }


def combine_training_frames(first_payload, second_payload):
    return {
        "X": pd.concat([first_payload["X"], second_payload["X"]], axis=0).reset_index(drop=True),
        "y": pd.concat([first_payload["y"], second_payload["y"]], axis=0).reset_index(drop=True),
        "meta": pd.concat([first_payload["meta"], second_payload["meta"]], axis=0).reset_index(
            drop=True
        )
        if first_payload["meta"] is not None and second_payload["meta"] is not None
        else None,
    }


def _extract_partition_payload(df, feature_columns, target_col, meta_columns):
    return {
        "X": df[feature_columns].copy(),
        "y": df[target_col].copy(),
        "meta": df[meta_columns].copy() if meta_columns else None,
        "df": df.copy(),
    }


def _sample_partition(df, target_col, sample_frac, random_state):
    if sample_frac >= 1.0:
        return df.reset_index(drop=True)
    if df.empty:
        return df.reset_index(drop=True)

    sample_size = max(1, int(round(len(df) * sample_frac)))
    if sample_size >= len(df):
        return df.reset_index(drop=True)

    if df[target_col].nunique() > 1 and sample_size > 1:
        sampled, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[target_col],
            random_state=random_state,
        )
        return sampled.sort_values("issue_date").reset_index(drop=True)

    return (
        df.sample(n=sample_size, random_state=random_state)
        .sort_values("issue_date")
        .reset_index(drop=True)
    )


def _summarize_partition(name, df, target_col, date_col):
    if df.empty:
        return {
            "partition": name,
            "rows": 0,
            "default_rate": float("nan"),
            "min_issue_date": pd.NaT,
            "max_issue_date": pd.NaT,
        }

    return {
        "partition": name,
        "rows": int(len(df)),
        "default_rate": float(df[target_col].mean()),
        "min_issue_date": df[date_col].min(),
        "max_issue_date": df[date_col].max(),
    }
