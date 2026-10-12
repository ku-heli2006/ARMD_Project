"""
Step: Does adding network features improve prediction?
-----------------------------------------------------------
Trains XGBoost twice on the EXACT same train/test split:
  1. WITHOUT network features (clinical/AMR/temporal only)
  2. WITH network features (+ network_degree, network_degree_centrality,
     network_cluster_size)

This directly tests your research question: does patient-contact network
information improve outbreak prediction over clinical data alone?

Input:  data/armd_with_network_features.parquet
Output: prints before/after comparison table
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

INPUT_PATH = "data/armd_with_network_features.parquet"
RANDOM_SEED = 42

NETWORK_COLS = ["network_degree", "network_degree_centrality", "network_cluster_size"]


def evaluate(name, y_test, y_pred, y_proba):
    return {
        "Model": name,
        "Precision (Outbreak)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (Outbreak)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 (Outbreak)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def train_and_eval(name, X, y, X_train_idx, X_test_idx):
    X_train, X_test = X.loc[X_train_idx], X.loc[X_test_idx]
    y_train, y_test = y.loc[X_train_idx], y.loc[X_test_idx]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200, scale_pos_weight=pos_weight,
        random_state=RANDOM_SEED, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return evaluate(name, y_test, y_pred, y_proba), model


def main():
    print("Loading dataset with network features...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Shape: {df.shape}")

    target_col = "outbreak_flag"
    y = df[target_col]

    id_and_meta_cols = [
        "anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded",
        "order_time_jittered_utc", "anon_id_ward", "pat_enc_csn_id_coded_ward",
        "order_time_jittered_utc_ward", "ward_type", "organism",
        "culture_description", target_col
    ]
    id_and_meta_cols = [c for c in id_and_meta_cols if c in df.columns]
    X_full = df.drop(columns=id_and_meta_cols)
    X_full = X_full.select_dtypes(include=[np.number])

    id_like_cols = [c for c in X_full.columns if
                    "anon_id" in c.lower() or "pat_enc_csn_id" in c.lower() or "order_proc_id" in c.lower()]
    X_full = X_full.drop(columns=id_like_cols)

    # Use the SAME split (by index) for both versions, so the comparison is fair
    X_train_idx, X_test_idx = train_test_split(
        X_full.index, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    present_network_cols = [c for c in NETWORK_COLS if c in X_full.columns]
    print(f"\nNetwork columns found: {present_network_cols}")

    # --- Version A: WITHOUT network features ---
    X_without = X_full.drop(columns=present_network_cols)
    result_without, model_without = train_and_eval(
        "WITHOUT network features", X_without, y, X_train_idx, X_test_idx
    )

    # --- Version B: WITH network features ---
    result_with, model_with = train_and_eval(
        "WITH network features", X_full, y, X_train_idx, X_test_idx
    )

    # --- Comparison table ---
    results_df = pd.DataFrame([result_without, result_with]).set_index("Model").round(4)
    print("\n" + "=" * 60)
    print("DOES NETWORK DATA IMPROVE PREDICTION?")
    print("=" * 60)
    print(results_df)

    improvement = results_df.loc["WITH network features"] - results_df.loc["WITHOUT network features"]
    print("\nChange (WITH minus WITHOUT):")
    print(improvement.round(4))

    # --- How important are the network features specifically? ---
    importances = pd.Series(model_with.feature_importances_, index=X_full.columns)
    network_importance = importances[present_network_cols].sort_values(ascending=False)
    print(f"\nNetwork feature importance (in the WITH-network model):")
    print(network_importance)
    print(f"\nNetwork features rank in top {(importances.rank(ascending=False)[present_network_cols] <= 20).sum()} of top 20 overall" if len(present_network_cols) else "")


if __name__ == "__main__":
    main()