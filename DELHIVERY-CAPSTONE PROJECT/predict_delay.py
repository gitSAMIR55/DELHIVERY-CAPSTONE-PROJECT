"""Part D: Scoring script for new incoming shipments."""
import os
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load inputs and artifacts
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    model = joblib.load("models/delay_model.pkl")
    threshold_df = pd.read_csv("models/delay_threshold.csv")
    threshold = float(threshold_df["threshold"].iloc[0])

    feature_order_df = pd.read_csv("models/feature_order.csv")
    feature_order = list(feature_order_df["feature"])

    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"].astype(float)))

    # 2. Clean shipments with stored medians
    clean_df, _ = clean_shipments(new_raw, medians=medians)

    # 3. Build features
    features_df = build_features(clean_df, hubs)

    # 4. Predict probabilities with feature columns in exact order
    X_new = features_df[feature_order]
    delay_probs = model.predict_proba(X_new)[:, 1]

    features_df["delay_probability"] = [round(float(p), 4) for p in delay_probs]
    features_df["flagged"] = (features_df["delay_probability"] >= threshold).astype(int)

    # 5. Join hub segment
    merged = features_df.merge(
        hub_segments[["hub_id", "segment"]], on="hub_id", how="left"
    )

    # 6. Assign action
    # Escalate to hub manager when flagged and segment is Escalate
    # Proactive customer call when flagged otherwise
    # No action when not flagged
    def assign_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            else:
                return "Proactive customer call"
        return "No action"

    merged["action"] = merged.apply(assign_action, axis=1)

    # Preserve file order of new_shipments.csv
    out_cols = [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
    result_df = merged[out_cols]
    result_df.to_csv("outputs/new_shipment_actions.csv", index=False)
    print(f"outputs/new_shipment_actions.csv written with {len(result_df)} rows.")


if __name__ == "__main__":
    main()
