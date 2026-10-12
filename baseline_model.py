"""
Step 5: Baseline ML Model (Logistic Regression)
---------------------------------------------------
Predicts outbreak_flag from the engineered features.

Because outbreak_flag is rare (~1.12%), we:
  - Use class_weight='balanced' so the model doesn't just predict "0" always
  - Report precision/recall/F1 for the outbreak class specifically, NOT
    just accuracy (accuracy would be misleadingly high and meaningless here)
  - Use stratified train/test split so both sets have a similar outbreak rate

Input:  data/armd_labeled.parquet
Output: prints evaluation results to the terminal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

INPUT_PATH = "data/armd_labeled.parquet"
RANDOM_SEED = 42


def main():
    print("Loading labeled dataset...")
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

    X = df.drop(columns=id_and_meta_cols)
    X = X.select_dtypes(include=[np.number])
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features used (first 10): {X.columns.tolist()[:10]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train outbreak rate: {y_train.mean() * 100:.2f}%")
    print(f"Test outbreak rate: {y_test.mean() * 100:.2f}%")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print("\nClassification Report (class 1 = outbreak):")
    print(classification_report(y_test, y_pred, target_names=["No Outbreak", "Outbreak"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("(rows = actual, columns = predicted; order: [No Outbreak, Outbreak])")

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {auc:.4f}")
    print("(0.5 = no better than random guessing, 1.0 = perfect separation)")

    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    print("\nTop 10 features by coefficient magnitude (simple explainability):")
    print(coefs.head(10))


if __name__ == "__main__":
    main()