"""
Step 2: Define the outbreak/HAI label
-----------------------------------------
Rule: A ward-day is flagged as a potential outbreak signal if 3+ positive
cultures with the SAME organism occur in the SAME ward within a 7-day window.

This is a working definition for research purposes -- not a clinical
diagnosis of an actual outbreak. Document this clearly in your report.

Input:  data/armd_sample_clean.parquet
Output: data/armd_labeled.parquet (same data + new 'outbreak_flag' column)
"""

import pandas as pd

INPUT_PATH = "data/armd_sample_clean.parquet"
OUTPUT_PATH = "data/armd_labeled.parquet"

WINDOW_DAYS = 7
MIN_CASES = 3


def main():
    print("Loading cleaned sample...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Shape: {df.shape}")

    # Print available columns to confirm what we actually have for this step
    ward_cols = [c for c in df.columns if "ward" in c.lower() or "hosp_ward" in c.lower()]
    print(f"\nWard-related columns found: {ward_cols}")

    time_col = "order_time_jittered_utc"
    if time_col not in df.columns:
        print(f"WARNING: {time_col} not found. Cannot build time-based label without a timestamp.")
        return

    if "was_positive" not in df.columns or "organism" not in df.columns:
        print("WARNING: 'was_positive' or 'organism' column not found. Check your columns list above.")
        return

    # --- Convert time column to actual datetime ---
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    if len(ward_cols) == 0:
        print("No ward column found -- cannot group by ward. Check available columns above.")
        return

    # Ward info comes as separate yes/no columns (hosp_ward_IP, hosp_ward_ICU, etc.)
    # rather than one single ward-name column. Combine them into one ward_type
    # column, prioritizing ICU > ER > IP > OP if a row has more than one flag set.
    ward_binary_cols = [c for c in ["hosp_ward_ICU", "hosp_ward_ER", "hosp_ward_IP", "hosp_ward_OP"]
                        if c in df.columns]
    print(f"Using binary ward columns: {ward_binary_cols}")

    def get_ward_type(row):
        for col in ward_binary_cols:  # already in priority order
            if row.get(col) == 1:
                return col.replace("hosp_ward_", "")
        return "UNKNOWN"

    df["ward_type"] = df.apply(get_ward_type, axis=1)
    print(f"Ward type distribution:\n{df['ward_type'].value_counts()}")
    ward_col = "ward_type"

    # --- Only positive cultures count toward outbreak signal ---
    # (created AFTER ward_type so it's actually included)
    positive = df[df["was_positive"] == 1].copy()  # adjust if was_positive uses different encoding (e.g. True/False)
    print(f"\nPositive culture rows: {len(positive)} out of {len(df)}")

    # --- Rolling count: for each (ward, organism) pair, count cases in a 7-day window ---
    positive = positive.sort_values(time_col)
    outbreak_orders = set()

    for (ward, organism), group in positive.groupby([ward_col, "organism"]):
        group = group.sort_values(time_col)
        times = group[time_col].values
        order_ids = group["order_proc_id_coded"].values

        # Sliding window check: for each case, count how many others fall within
        # WINDOW_DAYS after it
        for i in range(len(times)):
            window_end = times[i] + pd.Timedelta(days=WINDOW_DAYS)
            count_in_window = ((times >= times[i]) & (times <= window_end)).sum()
            if count_in_window >= MIN_CASES:
                # Flag all orders involved in this window
                involved = order_ids[(times >= times[i]) & (times <= window_end)]
                outbreak_orders.update(involved)

    print(f"\nFlagged {len(outbreak_orders)} culture orders as part of a potential outbreak signal")

    df["outbreak_flag"] = df["order_proc_id_coded"].isin(outbreak_orders).astype(int)
    print(f"\nLabel distribution:")
    print(df["outbreak_flag"].value_counts())
    print(f"Outbreak rate: {df['outbreak_flag'].mean() * 100:.2f}%")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved labeled dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()