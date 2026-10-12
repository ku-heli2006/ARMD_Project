"""
Step 5b: Model Comparison -- Logistic Regression vs Random Forest vs XGBoost
--------------------------------------------------------------------------------
Same data, same train/test split as the baseline, so results are directly
comparable. This answers your core research question: does a more complex
model actually do better than a simple one on this data?

Input:  data/armd_labeled.parquet
Output: prints comparison table to the terminal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("NOTE: xgboost not installed. Run 'pip install xgboost' to include it in this comparison.")

INPUT_PATH = "data/armd_labeled.parquet"
RANDOM_SEED = 42


def evaluate(name, y_test, y_pred, y_proba):
    return {
        "Model": name,
        "Precision (Outbreak)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (Outbreak)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 (Outbreak)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


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

    # Extra safety: drop ANY remaining column that looks like a leftover ID
    # (these carry no real clinical signal and can distort feature importance)
    id_like_cols = [c for c in X.columns if
                    "anon_id" in c.lower() or
                    "pat_enc_csn_id" in c.lower() or
                    "order_proc_id" in c.lower()]
    if id_like_cols:
        print(f"Dropping leftover ID-like columns: {id_like_cols}")
        X = X.drop(columns=id_like_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    results = []

    # --- Model A: Logistic Regression (baseline, needs scaling) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    logreg.fit(X_train_scaled, y_train)
    y_pred = logreg.predict(X_test_scaled)
    y_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, y_pred, y_proba))

    # --- Model B: Random Forest (doesn't need scaling, handles nonlinearity) ---
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, y_pred, y_proba))

    # --- Model C: XGBoost (if installed) ---
    if HAS_XGBOOST:
        # scale_pos_weight compensates for class imbalance, XGBoost's equivalent
        # of class_weight='balanced'
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200, scale_pos_weight=pos_weight,
            random_state=RANDOM_SEED, eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        y_proba = xgb.predict_proba(X_test)[:, 1]
        results.append(evaluate("XGBoost", y_test, y_pred, y_proba))

    # --- Comparison table ---
    results_df = pd.DataFrame(results).set_index("Model").round(4)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df)

    best_model = results_df["ROC-AUC"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_model}")

    # --- Feature importance from Random Forest (more reliable than logistic coefficients) ---
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 features by Random Forest importance:")
    print(importances.head(10))


if __name__ == "__main__":
    main()