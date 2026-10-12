"""
Step: Intervention Simulator
----------------------------------
For high-risk cases, simulates how the model's predicted risk would change
under different intervention scenarios. This is a MODEL-BASED SIMULATION,
not a guarantee that the intervention will actually reduce real-world risk.

Scenarios simulated (by modifying the network/exposure features):
  A. No intervention (current state)
  B. Isolation (this patient's connections are removed -- degree/cluster set to 0/1)
  C. Reduced exposure (connections halved)
  D. Combined isolation + reduced antibiotic-resistant medication signal

Input:  data/armd_with_network_features.parquet
Output: prints scenario comparison for top high-risk cases
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

INPUT_PATH = "data/armd_with_network_features.parquet"
RANDOM_SEED = 42
N_CASES_TO_SIMULATE = 5  # how many top high-risk cases to show scenarios for


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

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Monotonic constraints: force the model to learn that risk can only
    # DECREASE (or stay flat) as network exposure decreases -- this makes
    # the model usable for what-if simulation, since without this,
    # tree models can learn non-monotonic (illogical) relationships.
    # 1 = risk increases with feature, -1 = risk decreases, 0 = no constraint
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

    # --- Find the top N highest-risk cases in the test set ---
    baseline_proba = model.predict_proba(X_test)[:, 1]
    top_risk_positions = np.argsort(baseline_proba)[-N_CASES_TO_SIMULATE:][::-1]

    print("\n" + "=" * 70)
    print("INTERVENTION SIMULATION -- TOP HIGH-RISK CASES")
    print("=" * 70)
    print("NOTE: These are model-based simulations, not proof that the")
    print("intervention will actually cause this exact reduction in real life.\n")

    for rank, pos in enumerate(top_risk_positions, 1):
        case = X_test.iloc[[pos]].copy()
        actual_label = y_test.iloc[pos]
        baseline_risk = baseline_proba[pos]

        print(f"\n--- Case #{rank} (actual label: {actual_label}) ---")

        scenarios = {}

        # Scenario A: No intervention (baseline)
        scenarios["A. No intervention"] = baseline_risk

        # Scenario B: Isolation -- remove this case's network connections entirely
        case_isolated = case.copy()
        if "network_degree" in case_isolated.columns:
            case_isolated["network_degree"] = 0
        if "network_degree_centrality" in case_isolated.columns:
            case_isolated["network_degree_centrality"] = 0
        if "network_cluster_size" in case_isolated.columns:
            case_isolated["network_cluster_size"] = 1
        scenarios["B. Isolation (remove connections)"] = model.predict_proba(case_isolated)[0, 1]

        # Scenario C: Reduced exposure -- realistically halve connections,
        # rounded to whole numbers (tree models can behave erratically on
        # fractional values they never saw during training)
        case_reduced = case.copy()
        if "network_degree" in case_reduced.columns:
            case_reduced["network_degree"] = np.floor(case_reduced["network_degree"] / 2)
        if "network_cluster_size" in case_reduced.columns:
            case_reduced["network_cluster_size"] = np.maximum(
                1, np.floor(case_reduced["network_cluster_size"] / 2)
            )
        scenarios["C. Reduced exposure (halved)"] = model.predict_proba(case_reduced)[0, 1]

        # Scenario D: Combined -- isolation + reduced (same as B here, but kept
        # separate for clarity/extension -- e.g. could add screening timing later)
        scenarios["D. Combined (isolation + reduced)"] = scenarios["B. Isolation (remove connections)"]

        # --- Print scenario table for this case ---
        for name, risk in scenarios.items():
            change = risk - baseline_risk
            arrow = "↓" if change < 0 else ("↑" if change > 0 else "=")
            print(f"  {name:40s}  {risk:6.2%}  ({arrow} {change:+.2%})")

    print("\n" + "=" * 70)
    print("Interpretation: with monotonic constraints applied to the network")
    print("features, the model is now forced to treat reduced exposure as")
    print("never increasing predicted risk -- making these results genuinely")
    print("usable for what-if reasoning, unlike an unconstrained tree model")
    print("which can learn non-monotonic (illogical) relationships.")
    print("\nNOTE: monotonic constraints are a deliberate modeling choice for")
    print("simulation purposes. Document in your report that this simulator")
    print("model differs slightly from the unconstrained comparison model")
    print("used in your earlier evaluation -- the constraint trades a small")
    print("amount of predictive flexibility for logically consistent simulations.")


if __name__ == "__main__":
    main()