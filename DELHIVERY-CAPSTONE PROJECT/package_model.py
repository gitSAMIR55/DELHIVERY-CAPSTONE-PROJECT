"""Part D: Package final classifier and models artifacts."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import src.shipment_prep as sp


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Read config from Part C
    cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    selector_k = int(cfg["selector_k"])
    n_estimators = int(cfg["n_estimators"])
    max_depth = None if str(cfg["max_depth"]).lower() == "unlimited" else int(cfg["max_depth"])
    min_samples_leaf = int(cfg["min_samples_leaf"])
    threshold = float(cfg["threshold"])

    # 2. Rebuild Part C's stratified split
    df = pd.read_csv("data/modelling_table.csv")
    X = df[sp.FEATURES]
    y = df["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 3. Fit final pipeline on training rows
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("model", rf),
    ])
    pipeline.fit(X_train, y_train)

    # 4. Save artifacts in models/
    # Model pickle
    joblib.dump(pipeline, "models/delay_model.pkl")

    # Threshold CSV
    pd.DataFrame([{"threshold": round(threshold, 4)}]).to_csv(
        "models/delay_threshold.csv", index=False
    )

    # Feature order CSV
    pd.DataFrame({"feature": sp.FEATURES}).to_csv(
        "models/feature_order.csv", index=False
    )

    # Impute values CSV: medians from clean_shipments on historical data
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = sp.clean_shipments(raw)
    impute_df = pd.DataFrame([
        {"column": k, "median": round(float(v), 4)} for k, v in medians.items()
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)

    print("Artifacts saved to models/ directory.")

    # 5. Reload verification on test rows
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    probs_in_memory = pipeline.predict_proba(X_test)[:, 1]
    probs_reloaded = reloaded_pipeline.predict_proba(X_test)[:, 1]

    reload_max_diff = float(np.max(np.abs(probs_in_memory - probs_reloaded)))

    reload_check = [
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", round(reload_max_diff, 8)),
        ("threshold", round(threshold, 4)),
    ]
    pd.DataFrame(reload_check, columns=["metric", "value"]).to_csv(
        "outputs/part_d_reload_check.csv", index=False
    )
    print("outputs/part_d_reload_check.csv written.")


if __name__ == "__main__":
    main()
