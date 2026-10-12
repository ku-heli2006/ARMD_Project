"""
ARMD (Antibiotic Resistance Microbiology Dataset) — Load & Join Script
------------------------------------------------------------------------
Purpose: Load the ARMD files, join them on order_proc_id_coded, and
pivot the long-format tables into wide engineered features.

STATUS:
  - Cohort, demographics, ward info, resistance, prior organism: WORKING
  - Labs, vitals, antibiotic exposure, prior meds, prior procedures: PLACEHOLDERS
    (need to check exact column names before filling in the pivot logic)
  - Comorbidity: SKIPPED (19.67 GB file, excluded by choice -- see project notes)
"""

import pandas as pd
import os

DATA_DIR = "./data"  # <-- change this if your CSVs are somewhere else

# Join keys used across every file in this dataset
KEY_ORDER = "order_proc_id_coded"      # one row = one culture order (finest grain)
KEY_ENC   = "pat_enc_csn_id_coded"     # encounter
KEY_PAT   = "anon_id"                  # patient
KEY_TIME  = "order_time_jittered_utc"  # jittered timestamp


def load_csv(filename, **kwargs):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  [skip] {filename} not found at {path}")
        return None
    print(f"  [load] {filename}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def main():
    print("Loading core files...")

    # --- 1. Core cohort table ---
    # This file is one row PER ANTIBIOTIC TESTED per culture, not one row
    # per culture order. We collapse it to one row per order first, or
    # every later merge will multiply rows together (memory crash).
    cohort_raw = load_csv("microbiology_cultures_cohort.csv")
    if cohort_raw is None:
        raise SystemExit("cohort.csv is required as the base table. Download it first.")

    print(f"Raw cohort shape (long format): {cohort_raw.shape}")
    n_unique_orders = cohort_raw[KEY_ORDER].nunique()
    print(f"Unique culture orders: {n_unique_orders}")

    id_cols = [KEY_PAT, KEY_ENC, KEY_ORDER]
    other_cols = [c for c in ["culture_description", "was_positive", "organism", KEY_TIME]
                  if c in cohort_raw.columns]
    base_info = cohort_raw[id_cols + other_cols].drop_duplicates(subset=KEY_ORDER)

    if "antibiotic" in cohort_raw.columns and "susceptibility" in cohort_raw.columns:
        susceptibility_wide = cohort_raw.pivot_table(
            index=KEY_ORDER,
            columns="antibiotic",
            values="susceptibility",
            aggfunc="first"
        )
        susceptibility_wide.columns = [f"SUSC__{c}" for c in susceptibility_wide.columns]
        print(f"  Susceptibility pivoted to {susceptibility_wide.shape[1]} columns")
        cohort = base_info.merge(susceptibility_wide, on=KEY_ORDER, how="left")
    else:
        cohort = base_info

    # Sanity check: must be unique now, or every later merge explodes again
    n_dupes = cohort[KEY_ORDER].duplicated().sum()
    if n_dupes > 0:
        raise SystemExit(
            f"ERROR: {n_dupes} duplicate {KEY_ORDER} values remain after collapsing cohort."
        )

    print(f"Base cohort shape (one row per order): {cohort.shape}")

    # --- 2. Demographics: one row per patient ---
    demo = load_csv("microbiology_cultures_demographics.csv")
    if demo is not None:
        demo = demo.drop_duplicates(subset=KEY_PAT)
        cohort = cohort.merge(demo, on=[KEY_PAT], how="left", suffixes=("", "_demo"), validate="m:1")
        print(f"  After demographics merge: {cohort.shape}")

    # --- 3. Ward info: one row per culture order ---
    ward = load_csv("microbiology_cultures_ward_info.csv")
    if ward is not None:
        ward = ward.drop_duplicates(subset=KEY_ORDER)
        cohort = cohort.merge(ward, on=[KEY_ORDER], how="left", suffixes=("", "_ward"), validate="m:1")
        print(f"  After ward info merge: {cohort.shape}")

    # --- 4. Microbial resistance: organism x antibiotic per culture -> pivot ---
    resistance = load_csv("microbiology_cultures_microbial_resistance.csv")
    if resistance is not None:
        resistance["organism_antibiotic"] = (
            resistance["organism"].astype(str) + "__" + resistance["antibiotic"].astype(str)
        )
        resistance_wide = resistance.pivot_table(
            index=KEY_ORDER,
            columns="organism_antibiotic",
            values="resistant_time_to_culturetime",
            aggfunc="min"
        )
        resistance_wide.columns = [f"AMR__{c}" for c in resistance_wide.columns]
        print(f"  Resistance pivoted to {resistance_wide.shape[1]} columns")
        cohort = cohort.merge(resistance_wide, on=KEY_ORDER, how="left", validate="m:1")
        print(f"  After resistance merge: {cohort.shape}")

    # --- 5. Prior infecting organism -> pivot ---
    # NOTE: the source file has a typo in one column name ("culutre" vs "culture").
    # This code checks for both spellings automatically.
    prior_org = load_csv("microbiology_culture_prior_infecting_organism.csv")
    if prior_org is not None and "prior_organism" in prior_org.columns:
        value_col = "prior_infecting_organism_days_to_culutre"
        if value_col not in prior_org.columns:
            value_col = "prior_infecting_organism_days_to_culture"
        prior_org_wide = prior_org.pivot_table(
            index=KEY_ORDER,
            columns="prior_organism",
            values=value_col,
            aggfunc="min"
        )
        prior_org_wide.columns = [f"PRIOR_ORG__{c}" for c in prior_org_wide.columns]
        cohort = cohort.merge(prior_org_wide, on=KEY_ORDER, how="left", validate="m:1")
        print(f"  After prior organism merge: {cohort.shape}")

    # =========================================================================
    # PLACEHOLDER SECTIONS -- not yet implemented, waiting on column-name check
    # =========================================================================

    # --- 6. Labs: one row per (order, Period_Day) -> pivot on stat AND period ---
    labs = load_csv("microbiology_cultures_labs.csv")
    if labs is not None:
        labs_id_cols = [KEY_PAT, KEY_ENC, KEY_ORDER, "Period_Day"]
        labs_value_cols = [c for c in labs.columns if c not in labs_id_cols]
        labs_wide = labs.pivot_table(
            index=KEY_ORDER,
            columns="Period_Day",
            values=labs_value_cols,
            aggfunc="first"
        )
        # Flatten the two-level column names (stat_name, period_day) -> "LAB__stat_dayN"
        labs_wide.columns = [f"LAB__{val}_day{period}" for val, period in labs_wide.columns]
        print(f"  Labs pivoted to {labs_wide.shape[1]} columns")
        cohort = cohort.merge(labs_wide, on=KEY_ORDER, how="left", validate="m:1")
        print(f"  After labs merge: {cohort.shape}")

    # --- 7. Vitals: already one row per order (no Period_Day column) -> merge directly ---
    vitals = load_csv("microbiology_cultures_vitals.csv")
    if vitals is not None:
        vitals = vitals.drop_duplicates(subset=KEY_ORDER)  # safety: force one row per order
        vitals_cols_to_use = [c for c in vitals.columns if c == KEY_ORDER or c not in [KEY_PAT, KEY_ENC]]
        vitals_renamed = vitals[vitals_cols_to_use].copy()
        rename_map = {c: f"VITAL__{c}" for c in vitals_renamed.columns if c != KEY_ORDER}
        vitals_renamed = vitals_renamed.rename(columns=rename_map)
        print(f"  Vitals: {vitals_renamed.shape[1] - 1} columns")
        cohort = cohort.merge(vitals_renamed, on=KEY_ORDER, how="left", validate="m:1")
        print(f"  After vitals merge: {cohort.shape}")

    # --- 8. Antibiotic class exposure: one row per (order, class) -> pivot ---
    abx_class = load_csv("microbiology_cultures_antibiotic_class_exposure.csv")
    if abx_class is not None and "antibiotic_class" in abx_class.columns:
        abx_class_wide = abx_class.pivot_table(
            index=KEY_ORDER,
            columns="antibiotic_class",
            values="time_to_culturetime",
            aggfunc="min"
        )
        abx_class_wide.columns = [f"ABXCLASS__{c}" for c in abx_class_wide.columns]
        print(f"  Antibiotic class exposure pivoted to {abx_class_wide.shape[1]} columns")
        cohort = cohort.merge(abx_class_wide, on=KEY_ORDER, how="left", validate="m:1")
        print(f"  After antibiotic class exposure merge: {cohort.shape}")

    # --- 9. Antibiotic subtype exposure: one row per (order, subtype) -> pivot ---
    abx_subtype = load_csv("microbiology_cultures_antibiotic_subtype_exposure.csv")
    if abx_subtype is not None and "antibiotic_subtype" in abx_subtype.columns:
        value_col = "medication_time_to_cultureTime"
        if value_col not in abx_subtype.columns:
            value_col = "time_to_culturetime"  # fallback to README-documented name
        abx_subtype_wide = abx_subtype.pivot_table(
            index=KEY_ORDER,
            columns="antibiotic_subtype",
            values=value_col,
            aggfunc="min"
        )
        abx_subtype_wide.columns = [f"ABXSUBTYPE__{c}" for c in abx_subtype_wide.columns]
        print(f"  Antibiotic subtype exposure pivoted to {abx_subtype_wide.shape[1]} columns")
        cohort = cohort.merge(abx_subtype_wide, on=KEY_ORDER, how="left", validate="m:1")
        print(f"  After antibiotic subtype exposure merge: {cohort.shape}")

    # --- 10. Prior medications: one row per (order, medication) -> pivot ---
    prior_med = load_csv("microbiology_cultures_prior_med.csv")
    if prior_med is not None and "medication_name" in prior_med.columns:
        value_col = "medication_time_to_culturetime"
        if value_col not in prior_med.columns:
            for candidate in ["medication_time_to_cultureTime", "time_to_culturetime"]:
                if candidate in prior_med.columns:
                    value_col = candidate
                    break
        prior_med_wide = prior_med.pivot_table(
            index=KEY_ORDER,
            columns="medication_name",
            values=value_col,
            aggfunc="min"
        )
        prior_med_wide.columns = [f"MED__{c}" for c in prior_med_wide.columns]
        print(f"  Prior medications pivoted to {prior_med_wide.shape[1]} columns")
        cohort = cohort.merge(prior_med_wide, on=KEY_ORDER, how="left", validate="m:1")
        print(f"  After prior medications merge: {cohort.shape}")

    # --- 11. Prior procedures: one row per (order, procedure) -> pivot ---
    prior_proc = load_csv("microbiology_cultures_priorprocedures.csv")
    proc_name_col = None
    if prior_proc is not None:
        for candidate in ["procedure_name", "procedure_description"]:
            if candidate in prior_proc.columns:
                proc_name_col = candidate
                break
    if prior_proc is not None and proc_name_col is not None:
        value_col = "procedure_time_to_culturetime"
        if value_col not in prior_proc.columns:
            for candidate in ["procedure_time_to_cultureTime", "time_to_culturetime"]:
                if candidate in prior_proc.columns:
                    value_col = candidate
                    break
        prior_proc_wide = prior_proc.pivot_table(
            index=KEY_ORDER,
            columns=proc_name_col,
            values=value_col,
            aggfunc="min"
        )
        prior_proc_wide.columns = [f"PROC__{c}" for c in prior_proc_wide.columns]
        print(f"  Prior procedures pivoted to {prior_proc_wide.shape[1]} columns")
        cohort = cohort.merge(prior_proc_wide, on=KEY_ORDER, how="left", validate="m:1")
        print(f"  After prior procedures merge: {cohort.shape}")

    # NOTE: comorbidity.csv (19.67 GB) intentionally excluded -- see project notes

    # =========================================================================

    print(f"\nFinal joined table shape: {cohort.shape}")
    print(f"  Rows (culture orders): {cohort.shape[0]}")
    print(f"  Columns (features so far): {cohort.shape[1]}")

    out_path = os.path.join(DATA_DIR, "armd_base_joined.parquet")
    cohort.to_parquet(out_path, index=False)
    print(f"\nSaved joined base table to: {out_path}")
    print("Use Parquet, not CSV, from here on -- it's much smaller and faster to reload.")


if __name__ == "__main__":
    main()