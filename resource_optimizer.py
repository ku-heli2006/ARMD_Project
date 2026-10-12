"""
Step: Resource-Constrained Optimizer
------------------------------------------
Given a limited number of isolation beds, decides WHICH patients to
isolate to maximize total predicted risk reduction -- using the same
monotonic-constrained model and isolation logic from the intervention
simulator.

Approach: GREEDY allocation. For every high-risk candidate, estimate the
risk reduction isolation would give them, then pick the top N candidates
by reduction, where N = number of beds available.

Input:  data/armd_with_network_features.parquet
Output: prints optimal allocation for a given bed count
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

INPUT_PATH = "data/armd_with_network_features.parquet"
RANDOM_SEED = 42

# --- Resource constraints (adjust these to match your scenario) ---
ISOLATION_BEDS_AVAILABLE = 5
N_CANDIDATES_TO_CONSIDER = 20  # look at the top 20 highest-risk cases as candidates


def main():
    print("Loading dataset...")
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
                    "anon_id" in c.lower() or "pat_enc_csn_id" in c.lower() or "order_proc_id" in c.lower()]
    X = X.drop(columns=id_like_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # --- Same monotonic-constrained model as the intervention simulator ---
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    monotone_map = {
        "network_degree": 1,
        "network_degree_centrality": 1,
        "network_cluster_size": 1,
    }
    constraints = tuple(monotone_map.get(col, 0) for col in X.columns)

    model = XGBClassifier(
        n_estimators=200, scale_pos_weight=pos_weight,
        random_state=RANDOM_SEED, eval_metric="logloss",
        monotone_constraints=constraints
    )
    model.fit(X_train, y_train)

    # --- Step 1: identify top N candidates by current predicted risk ---
    baseline_proba = model.predict_proba(X_test)[:, 1]
    top_candidates_positions = np.argsort(baseline_proba)[-N_CANDIDATES_TO_CONSIDER:][::-1]

    # --- Step 2: for each candidate, estimate risk reduction from isolation ---
    candidate_results = []
    for pos in top_candidates_positions:
        case = X_test.iloc[[pos]].copy()
        baseline_risk = baseline_proba[pos]

        case_isolated = case.copy()
        if "network_degree" in case_isolated.columns:
            case_isolated["network_degree"] = 0
        if "network_degree_centrality" in case_isolated.columns:
            case_isolated["network_degree_centrality"] = 0
        if "network_cluster_size" in case_isolated.columns:
            case_isolated["network_cluster_size"] = 1
        isolated_risk = model.predict_proba(case_isolated)[0, 1]

        risk_reduction = baseline_risk - isolated_risk

        candidate_results.append({
            "test_set_position": pos,
            "baseline_risk": baseline_risk,
            "risk_if_isolated": isolated_risk,
            "risk_reduction": risk_reduction,
        })

    candidates_df = pd.DataFrame(candidate_results).sort_values("risk_reduction", ascending=False)
    candidates_df = candidates_df.reset_index(drop=True)

    # --- Step 3: greedy allocation -- pick top N by risk reduction, where N = beds available ---
    print("\n" + "=" * 70)
    print(f"RESOURCE OPTIMIZER: {ISOLATION_BEDS_AVAILABLE} isolation beds available")
    print(f"Considering top {N_CANDIDATES_TO_CONSIDER} highest-risk candidates")
    print("=" * 70)

    print(f"\nAll {N_CANDIDATES_TO_CONSIDER} candidates ranked by risk reduction from isolation:")
    display_df = candidates_df.copy()
    display_df["baseline_risk"] = display_df["baseline_risk"].apply(lambda x: f"{x:.2%}")
    display_df["risk_if_isolated"] = display_df["risk_if_isolated"].apply(lambda x: f"{x:.2%}")
    display_df["risk_reduction"] = display_df["risk_reduction"].apply(lambda x: f"{x:.2%}")
    print(display_df.to_string(index=True))

    allocated = candidates_df.head(ISOLATION_BEDS_AVAILABLE)
    not_allocated = candidates_df.iloc[ISOLATION_BEDS_AVAILABLE:]

    total_reduction_allocated = allocated["risk_reduction"].sum()
    total_reduction_if_all = candidates_df["risk_reduction"].sum()
    avg_reduction_allocated = allocated["risk_reduction"].mean()
    avg_reduction_if_all = candidates_df["risk_reduction"].mean()

    print(f"\n--- OPTIMAL ALLOCATION ({ISOLATION_BEDS_AVAILABLE} beds) ---")
    print(f"Isolate candidates at positions: {allocated['test_set_position'].tolist()}")
    print(f"\nAverage risk reduction per isolated patient: {avg_reduction_allocated:.2%}")
    print(f"(Average across all {N_CANDIDATES_TO_CONSIDER} candidates if unlimited beds: {avg_reduction_if_all:.2%})")
    print(f"\nSum of individual reductions (allocated {ISOLATION_BEDS_AVAILABLE} beds): {total_reduction_allocated:.2f}")
    print(f"Sum of individual reductions (all {N_CANDIDATES_TO_CONSIDER} candidates): {total_reduction_if_all:.2f}")
    print(f"NOTE: these sums are a 'total risk-reduction units' score for comparing")
    print(f"allocations, NOT a percentage -- summing per-patient percentages does")
    print(f"not itself represent a single meaningful percentage.")
    print(f"\nEfficiency: this allocation of {ISOLATION_BEDS_AVAILABLE}/{N_CANDIDATES_TO_CONSIDER} beds captures")
    print(f"  {total_reduction_allocated / total_reduction_if_all * 100:.1f}% of the total reduction-units")
    print(f"  achievable if all {N_CANDIDATES_TO_CONSIDER} candidates could be isolated.")

    print("\nNOTE: This is a greedy allocation based on model-estimated individual")
    print("risk reduction. It does not account for interactions between isolating")
    print("multiple connected patients simultaneously (a more advanced version")
    print("could use integer programming to model those interactions).")


if __name__ == "__main__":
    main()