"""
Step 1: Sample down to ~10,000 rows + handle missing values
--------------------------------------------------------------
Input:  data/armd_base_joined.parquet   (751,075 rows x 1,506 columns)
Output: data/armd_sample_clean.parquet  (~10,000 rows, cleaned)
"""

import pandas as pd
import numpy as np

INPUT_PATH = "data/armd_base_joined.parquet"
OUTPUT_PATH = "data/armd_sample_clean.parquet"
SAMPLE_SIZE = 10000
RANDOM_SEED = 42  # fixed seed = reproducible sample (same result every time you run it)


def main():
    print("Loading full joined dataset...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Full dataset shape: {df.shape}")

    # --- 1. Random sample of ~10,000 rows ---
    # random_state fixes the seed so your sample is reproducible -- important
    # for your report, so anyone re-running this gets the same 10,000 rows.
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sampled shape: {df_sample.shape}")

    # --- 2. Check how sparse the data is ---
    missing_pct = df_sample.isna().mean() * 100
    print(f"\nMissingness overview:")
    print(f"  Columns with >95% missing: {(missing_pct > 95).sum()}")
    print(f"  Columns with >80% missing: {(missing_pct > 80).sum()}")
    print(f"  Columns with <20% missing: {(missing_pct < 20).sum()}")

    # --- 3. Drop columns that are almost entirely empty ---
    # A column that's 99%+ empty gives the model almost nothing to learn from.
    # Threshold is a judgment call -- 95% is a reasonable, defensible starting point.
    DROP_THRESHOLD = 95
    cols_to_drop = missing_pct[missing_pct > DROP_THRESHOLD].index.tolist()
    print(f"\nDropping {len(cols_to_drop)} columns that are >{DROP_THRESHOLD}% empty...")
    df_sample = df_sample.drop(columns=cols_to_drop)
    print(f"Shape after dropping near-empty columns: {df_sample.shape}")

    # --- 4. Fill remaining missing values ---
    # Most of your engineered columns (AMR__, MED__, PROC__, etc.) represent
    # "days to event" -- missing means "this never happened for this patient",
    # not "unknown". So we fill those with a sentinel value, not the mean.
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
    prefix_groups = ["AMR__", "PRIOR_ORG__", "ABXCLASS__", "ABXSUBTYPE__", "MED__", "PROC__"]
    event_cols = [c for c in numeric_cols if any(c.startswith(p) for p in prefix_groups)]
    other_numeric_cols = [c for c in numeric_cols if c not in event_cols]

    print(f"\nFilling {len(event_cols)} 'event' columns with -1 (meaning: did not occur)...")
    df_sample[event_cols] = df_sample[event_cols].fillna(-1)

    print(f"Filling {len(other_numeric_cols)} other numeric columns (labs/vitals) with column median...")
    for col in other_numeric_cols:
        df_sample[col] = df_sample[col].fillna(df_sample[col].median())

    # --- 5. Save ---
    df_sample.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned sample to: {OUTPUT_PATH}")
    print(f"Final shape: {df_sample.shape}")
    print(f"  Rows: {df_sample.shape[0]:,}")
    print(f"  Columns: {df_sample.shape[1]:,}")


if __name__ == "__main__":
    main()
    