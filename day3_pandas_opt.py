import numpy as np
import pandas as pd

def create_synthetic_data(n_rows: int = 1_000_000) -> pd.DataFrame:
    np.random.seed(42)
    cities = ["New York", "London", "Tokyo", "Berlin", "Bengaluru"]
    statuses = ["active", "pending", "failed", "completed"]

    return pd.DataFrame({
        "transaction_id": np.random.randint(1, 10000, size=n_rows),
        "age": np.random.randint(18, 90, size=n_rows),
        "amount": np.random.uniform(10.0, 5000.0, size=n_rows),
        "city": np.random.choice(cities, size=n_rows),
        "status": np.random.choice(statuses, size=n_rows),
    })

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_opt = df.copy()

    # Optimize numeric columns
    for col in df_opt.select_dtypes(include=["int", "integer"]).columns:
        if (df_opt[col] >= 0).all():
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="unsigned")
        else:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")

    for col in df_opt.select_dtypes(include=["float"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")

    # Convert low-cardinality strings to category
    for col in df_opt.select_dtypes(include=["object", "str"]).columns:
        num_unique = df_opt[col].nunique()
        num_total = len(df_opt[col])
        if (num_unique / num_total) < 0.2:
            df_opt[col] = df_opt[col].astype("category")

    return df_opt

def apply_vectorized_transforms(df: pd.DataFrame) -> pd.DataFrame:
    # Fast conditional mask
    high_value_mask = (df["amount"] > 2500.0) & (df["city"] == "Tokyo")

    # Explicit single-step mutation using .loc
    df.loc[high_value_mask, "priority_flag"] = 1
    df["priority_flag"] = df["priority_flag"].fillna(0).astype("uint8")
    return df

if __name__ == "__main__":
    df_raw = create_synthetic_data(1_000_000)
    raw_mem = df_raw.memory_usage(deep=True).sum() / (1024 ** 2)

    df_optimized = optimize_dataframe(df_raw)
    opt_mem = df_optimized.memory_usage(deep=True).sum() / (1024 ** 2)

    df_final = apply_vectorized_transforms(df_optimized)

    print(f"Raw DataFrame Memory:       {raw_mem:.2f} MB")
    print(f"Optimized DataFrame Memory: {opt_mem:.2f} MB")
    print(f"Total Memory Reduction:     {((raw_mem - opt_mem) / raw_mem) * 100:.2f}%")
    print("\nOptimized dtypes:")
    print(df_optimized.dtypes)
