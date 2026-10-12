"""
Step: Patient-Contact Network
----------------------------------
Builds a network of potential patient exposure based on overlapping
ward + time. Two culture orders are "connected" if they occurred in
the same ward within a defined time window of each other.

This gives us network features (exposure count, connections) that
were missing from the model -- directly motivated by the SHAP finding
that false positives lacked this context.

Input:  data/armd_labeled.parquet
Output: data/armd_with_network_features.parquet
"""

import pandas as pd
import numpy as np
import networkx as nx

INPUT_PATH = "data/armd_labeled.parquet"
OUTPUT_PATH = "data/armd_with_network_features.parquet"

OVERLAP_WINDOW_DAYS = 3  # how close in time two cultures need to be to count as "overlapping"


def main():
    print("Loading labeled dataset...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Shape: {df.shape}")

    time_col = "order_time_jittered_utc"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    ward_flag_cols = ["hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU"]
    ward_flag_cols = [c for c in ward_flag_cols if c in df.columns]

    def get_ward(row):
        for c in ward_flag_cols:
            if row[c] == 1:
                return c.replace("hosp_ward_", "")
        return "UNKNOWN"

    df["ward_type"] = df.apply(get_ward, axis=1)

    # --- Build the graph ---
    print("\nBuilding patient-contact graph (this may take a minute)...")
    G = nx.Graph()
    G.add_nodes_from(df["order_proc_id_coded"])

    OVERLAP_WINDOW_DAYS_LOCAL = 1  # tightened from 3 to 1 day -- same ward, same/next day is more realistic exposure
    MEANINGFUL_WARDS = ["IP", "ICU"]  # only build network within real inpatient settings, not OP/ER (too broad/high-volume)

    edge_count = 0
    for ward, group in df.groupby("ward_type"):
        if ward not in MEANINGFUL_WARDS:
            continue
        group = group.sort_values(time_col).reset_index(drop=True)
        times = group[time_col].values
        ids = group["order_proc_id_coded"].values

        # For each case, connect it to any other case in the same ward
        # within OVERLAP_WINDOW_DAYS_LOCAL
        for i in range(len(times)):
            window_end = times[i] + np.timedelta64(OVERLAP_WINDOW_DAYS_LOCAL, "D")
            connected_mask = (times > times[i]) & (times <= window_end)
            connected_ids = ids[connected_mask]
            for other_id in connected_ids:
                G.add_edge(ids[i], other_id)
                edge_count += 1

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Compute network features ---
    print("\nComputing network features (degree, clustering, connected components)...")
    degree_dict = dict(G.degree())

    # Connected components = potential clusters
    components = list(nx.connected_components(G))
    component_sizes = {node: len(comp) for comp in components for node in comp}

    # Degree centrality (normalized version of degree)
    degree_centrality = nx.degree_centrality(G)

    network_features = pd.DataFrame({
        "order_proc_id_coded": list(degree_dict.keys()),
        "network_degree": list(degree_dict.values()),
        "network_degree_centrality": [degree_centrality[k] for k in degree_dict.keys()],
        "network_cluster_size": [component_sizes.get(k, 1) for k in degree_dict.keys()],
    })

    print(f"\nNetwork feature summary:")
    print(network_features[["network_degree", "network_cluster_size"]].describe())

    # --- Merge back into main dataset ---
    df = df.merge(network_features, on="order_proc_id_coded", how="left")
    df["network_degree"] = df["network_degree"].fillna(0)
    df["network_degree_centrality"] = df["network_degree_centrality"].fillna(0)
    df["network_cluster_size"] = df["network_cluster_size"].fillna(1)

    # --- Report largest clusters (candidate outbreak clusters) ---
    cluster_sizes = pd.Series(component_sizes).value_counts().sort_index(ascending=False)
    print(f"\nLargest connected components (potential clusters):")
    top_components = sorted(components, key=len, reverse=True)[:5]
    for i, comp in enumerate(top_components, 1):
        print(f"  Cluster {i}: {len(comp)} connected culture orders")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved dataset with network features to: {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()