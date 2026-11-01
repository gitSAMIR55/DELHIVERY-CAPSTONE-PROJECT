"""Part A: Data Foundation and Delay Evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Read raw shipments and hubs
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    raw_rows = len(raw)

    # Dedup check before cleaning pipeline
    dedup = raw.drop_duplicates()
    duplicate_rows_removed = raw_rows - len(dedup)
    rows_after_dedup = len(dedup)

    service_type_spellings_before = dedup["service_type"].nunique()
    missing_distance_km_before = int(dedup["distance_km"].isna().sum())
    missing_parcel_weight_before = int(dedup["parcel_weight_kg"].isna().sum())
    sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())
    missing_declared_value_before = int(dedup["declared_value"].isna().sum())
    missing_payment_mode_before = int(dedup["payment_mode"].isna().sum())

    # 2. Clean shipments
    clean_df, medians = clean_shipments(raw)
    service_type_spellings_after = clean_df["service_type"].nunique()

    # 3. Build features
    modelling_table = build_features(clean_df, hubs)
    rows_after_hub_merge = len(modelling_table)

    # 4. Target variable: is_delayed (1 when delivery_hours > promised_hours, else 0)
    modelling_table["is_delayed"] = (
        modelling_table["delivery_hours"] > modelling_table["promised_hours"]
    ).astype(int)

    # Save data/modelling_table.csv
    modelling_table.to_csv("data/modelling_table.csv", index=False)
    print(f"Modelling table saved with {len(modelling_table)} rows.")

    # 5. Null checks in FEATURES + delivery_hours + is_delayed
    cols_to_check = FEATURES + ["delivery_hours", "is_delayed"]
    nulls_in_features_and_target = int(modelling_table[cols_to_check].isna().sum().sum())

    delayed_count = int(modelling_table["is_delayed"].sum())
    delay_rate = float(modelling_table["is_delayed"].mean())

    # Delivery hours percentiles and outliers (1.5 * IQR rule)
    hours = modelling_table["delivery_hours"]
    q1 = float(hours.quantile(0.25))
    q3 = float(hours.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((hours < lower_bound) | (hours > upper_bound)).sum())

    # Declared value statistics on filled column
    dec_val = modelling_table["declared_value_inr"]
    log_dec_val = modelling_table["log_declared_value"]

    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", round(medians["distance_km"], 4)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", round(delay_rate, 4)),
        ("hours_q1", round(q1, 4)),
        ("hours_q3", round(q3, 4)),
        ("hours_lower_bound", round(lower_bound, 4)),
        ("hours_upper_bound", round(upper_bound, 4)),
        ("hours_outlier_count", outlier_count),
        ("declared_value_mean_inr", round(float(dec_val.mean()), 4)),
        ("declared_value_median_inr", round(float(dec_val.median()), 4)),
        ("log_declared_value_mean", round(float(log_dec_val.mean()), 4)),
        ("log_declared_value_median", round(float(log_dec_val.median()), 4)),
    ]

    checks_df = pd.DataFrame(checks, columns=["metric", "value"])
    checks_df.to_csv("outputs/part_a_checks.csv", index=False)
    print("outputs/part_a_checks.csv written.")

    # 6. Delay evidence table: part_a_delay_by_group.csv
    modelling_table["distance_band"] = pd.cut(
        modelling_table["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
    )

    group_rows = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        grp = (
            modelling_table.groupby(dim, observed=False)
            .agg(
                shipments=("is_delayed", "count"),
                delayed=("is_delayed", "sum"),
                delay_rate=("is_delayed", "mean"),
                mean_hours=("delivery_hours", "mean"),
            )
            .reset_index()
        )
        grp.rename(columns={dim: "group"}, inplace=True)
        grp.insert(0, "dimension", dim)
        group_rows.append(grp)

    delay_by_group = pd.concat(group_rows, ignore_index=True)
    delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
    delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)
    delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    print("outputs/part_a_delay_by_group.csv written.")

    # 7. Correlations table: part_a_correlations.csv
    num_df = modelling_table.select_dtypes(include=["number"])
    corr_df = num_df.corr().round(4)
    corr_df.reset_index(names="feature", inplace=True)
    corr_df.to_csv("outputs/part_a_correlations.csv", index=False)
    print("outputs/part_a_correlations.csv written.")


if __name__ == "__main__":
    main()
