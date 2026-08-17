import numpy as np
import pandas as pd

def generate_raw_cohort_data(n_rows: int = 500_000) -> pd.DataFrame:
    np.random.seed(42)
    categories = ["Electronics", "Fashion", "Groceries", "Home"]
    regions = ["North", "South", "East", "West"]

    amounts = np.random.exponential(scale=100.0, size=n_rows)
    # Introduce 10% artificial missing values in transaction amount
    mask = np.random.rand(n_rows) < 0.10
    amounts[mask] = np.nan

    return pd.DataFrame({
        "category": np.random.choice(categories, size=n_rows),
        "region": np.random.choice(regions, size=n_rows),
        "amount": amounts,
        "latency_ms": np.random.randint(5, 500, size=n_rows),
    })

def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["category", "region"], as_index=False, dropna=False)
        .agg(
            mean_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            total_transactions=("amount", "size"),
            valid_transactions=("amount", "count"),
            p95_latency=("latency_ms", lambda x: np.percentile(x, 95)),
        )
    )
    return summary

def engineer_cohort_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    # Missingness indicator flag
    df_feat["amount_is_missing"] = df_feat["amount"].isna().astype("uint8")

    # Impute missing amounts using group median (category-level)
    group_medians = df_feat.groupby("category")["amount"].transform("median")
    df_feat["amount_imputed"] = df_feat["amount"].fillna(group_medians)

    # Compute cohort-normalized deviation feature
    group_means = df_feat.groupby("category")["amount_imputed"].transform("mean")
    group_stds = df_feat.groupby("category")["amount_imputed"].transform("std")
    
    df_feat["amount_zscore_by_category"] = (df_feat["amount_imputed"] - group_means) / group_stds
    return df_feat

if __name__ == "__main__":
    df_raw = generate_raw_cohort_data(500_000)
    
    summary_df = compute_summary_metrics(df_raw)
    print("Summary columns:", summary_df.columns.tolist())
    assert isinstance(summary_df.columns, pd.Index) and not isinstance(summary_df.columns, pd.MultiIndex)
    
    feat_df = engineer_cohort_features(df_raw)
    print(f"Remaining NaNs in amount_imputed: {feat_df['amount_imputed'].isna().sum()}")
    assert feat_df["amount_imputed"].isna().sum() == 0
    assert len(feat_df) == len(df_raw)
    
    print("Day 4 execution successful!")
