"""Part D: Hub Segments and Operations Playbook."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Hub profile from modelling table
    df = pd.read_csv("data/modelling_table.csv")
    hubs_df = pd.read_csv("data/hubs.csv")

    hub_profile = (
        df.groupby("hub_id", as_index=False)
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .sort_values("hub_id", ascending=True)
        .reset_index(drop=True)
    )

    # Round profile metrics for output
    hub_profile_out = hub_profile.copy()
    hub_profile_out["mean_delivery_hours"] = hub_profile_out["mean_delivery_hours"].round(4)
    hub_profile_out["delay_rate"] = hub_profile_out["delay_rate"].round(4)
    hub_profile_out["mean_distance_km"] = hub_profile_out["mean_distance_km"].round(4)
    hub_profile_out["mean_n_stops"] = hub_profile_out["mean_n_stops"].round(4)
    hub_profile_out.to_csv("outputs/hub_profile.csv", index=False)
    print("outputs/hub_profile.csv written.")

    # 2. Scaling for K-Means
    profile_features = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hub_profile[profile_features])

    # 3. Separation evidence for K in 2, 3, 4, 5
    dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
    n = len(X_scaled)
    sep_records = []

    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # Cluster sizes in descending order joined by hyphens
        counts = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
        cluster_sizes = "-".join(map(str, counts))

        # Across-to-within distance ratio
        within_dists = []
        across_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        across_over_within = float(np.mean(across_dists) / np.mean(within_dists))
        sep_records.append({
            "K": k,
            "across_over_within": round(across_over_within, 4),
            "cluster_sizes": cluster_sizes,
        })

    pd.DataFrame(sep_records).to_csv("outputs/part_d_separation.csv", index=False)
    print("outputs/part_d_separation.csv written.")

    # 4. K = 3 clustering and naming: Standard (lowest delay), Watch (middle), Escalate (highest)
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster"] = km3.fit_predict(X_scaled)

    cluster_delay = hub_profile.groupby("cluster")["delay_rate"].mean().sort_values()
    name_mapping = {
        cluster_delay.index[0]: "Standard",
        cluster_delay.index[1]: "Watch",
        cluster_delay.index[2]: "Escalate",
    }
    hub_profile["segment"] = hub_profile["cluster"].map(name_mapping)

    # Merge with hubs metadata
    merged = hub_profile.merge(hubs_df[["hub_id", "hub_city", "region", "hub_tier"]], on="hub_id")

    hub_segments = merged[["hub_id", "segment", "hub_city", "hub_tier"]].sort_values("hub_id")
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)
    print("outputs/hub_segments.csv written.")

    # Segment profile: means of hub-level values
    segment_profile = (
        merged.groupby("segment")
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .loc[["Standard", "Watch", "Escalate"]]
        .reset_index()
    )
    segment_profile["mean_delivery_hours"] = segment_profile["mean_delivery_hours"].round(4)
    segment_profile["delay_rate"] = segment_profile["delay_rate"].round(4)
    segment_profile["mean_distance_km"] = segment_profile["mean_distance_km"].round(4)
    segment_profile["mean_n_stops"] = segment_profile["mean_n_stops"].round(4)
    segment_profile.to_csv("outputs/segment_profile.csv", index=False)
    print("outputs/segment_profile.csv written.")

    # Tier vs segment mismatch
    # Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
    merged["expected_segment"] = merged["hub_tier"].map(tier_map)
    mismatch_df = merged[merged["segment"] != merged["expected_segment"]][
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].sort_values("hub_id")
    mismatch_df.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
    print("outputs/tier_vs_segment_mismatch.csv written.")


if __name__ == "__main__":
    main()
