"""
Step 6: Explainable AI (SHAP)
----------------------------------
Explains WHY the model flagged specific cases as high-risk, not just which
features matter overall (that's what feature_importances_ already gave us).

Uses XGBoost since it had the best F1 balance in the model comparison.

Input:  data/armd_labeled.parquet
Output: prints per-case explanations + saves a summary plot
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shap
import matplotlib
matplotlib.use("Agg")  # save plots to file instead of trying to open a window
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("XGBoost not installed. Run: pip install xgboost")

INPUT_PATH = "data/armd_labeled.parquet"
RANDOM_SEED = 42


def main():
    print("Loading labeled dataset...")
    df = pd.read_parquet(INPUT_PATH)

    target_col = "outbreak_flag"
    y = df[target_col]

    id_and_meta_cols = [
        "anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded",
        "order_time_jittered_utc", "anon_id_ward", "pat_enc_csn_id_coded_ward",
        "order_time_jittered_utc_ward", "ward_type", "organism",
        "culture_description", target_col
    ]
    id_and_meta_cols = [c for c in id_and_meta_cols if c in df.columns]
    X = df.drop(columns=id_and_meta_cols)
    X = X.select_dtypes(include=[np.number])

    id_like_cols = [c for c in X.columns if
                    "anon_id" in c.lower() or
                    "pat_enc_csn_id" in c.lower() or
                    "order_proc_id" in c.lower()]
    X = X.drop(columns=id_like_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # --- Train XGBoost (same setup as the comparison script) ---
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200, scale_pos_weight=pos_weight,
        random_state=RANDOM_SEED, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # --- SHAP explainer ---
    print("\nComputing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # --- 1. Global summary: which features matter most, and in which direction ---
    print("\nSaving global SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=150)
    plt.close()
    print("Saved: shap_summary_plot.png")

    # --- 2. Explain a few individual predictions ---
    # Find the test cases the model was most confident were outbreaks
    y_proba = model.predict_proba(X_test)[:, 1]
    top_risk_idx = np.argsort(y_proba)[-3:][::-1]  # top 3 highest predicted risk

    print("\n" + "=" * 60)
    print("EXPLAINING TOP 3 HIGHEST-RISK PREDICTIONS")
    print("=" * 60)

    for rank, idx in enumerate(top_risk_idx, 1):
        actual_label = y_test.iloc[idx]
        predicted_risk = y_proba[idx]
        print(f"\n--- Case #{rank} (predicted risk: {predicted_risk:.2%}, actual label: {actual_label}) ---")

        # Get this case's SHAP values, sorted by contribution magnitude
        case_shap = pd.Series(shap_values[idx], index=X_test.columns)
        top_contributors = case_shap.sort_values(key=abs, ascending=False).head(5)

        print("Top contributing factors:")
        for feature, value in top_contributors.items():
            direction = "increased" if value > 0 else "decreased"
            print(f"  {feature}: {direction} risk (SHAP value: {value:+.3f})")

    print("\nDone. Open shap_summary_plot.png to see the overall feature impact chart.")


if __name__ == "__main__":
    main()