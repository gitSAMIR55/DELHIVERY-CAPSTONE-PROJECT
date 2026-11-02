"""Shared shipment preparation and feature engineering pipeline for Delhivery Last-Mile Delay Intelligence."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw: pd.DataFrame, medians: dict = None):
    """
    Clean raw shipment export:
    1. drop exact duplicate rows
    2. service_type: strip and title case
    3. payment_mode: fillna with 'Unknown'
    4. declared_value: strip currency symbol and commas, convert to numeric declared_value_inr, drop original
    5. dispatch_time: convert to datetime
    6. parcel_weight_kg: > 100 sentinel treated as missing (NaN)
    7. fill missing distance_km, parcel_weight_kg, declared_value_inr with median
    Returns (clean_df, medians_dict).
    """
    df = raw.copy()
    # 1. drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. service_type: strip leading and trailing spaces and convert to title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # 3. payment_mode: replace missing value with 'Unknown'
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value: remove rupee sign and commas, convert to float, store in declared_value_inr, drop declared_value
    val_clean = df["declared_value"].astype(str).str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False).str.strip()
    # Handle NaNs which became 'nan' string
    val_numeric = pd.to_numeric(val_clean, errors="coerce")
    df["declared_value_inr"] = val_numeric
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. parcel_weight_kg: any value above 100 is sentinel -> NaN
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. fill missing with medians
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }
    else:
        # Ensure exact keys and float types
        medians = {k: float(medians[k]) for k in ["distance_km", "parcel_weight_kg", "declared_value_inr"]}

    for col, med_val in medians.items():
        df[col] = df[col].fillna(med_val)

    return df, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """
    Attach hub columns via left merge on hub_id:
    (hub_city, region, hub_tier, hub_daily_capacity)
    and create new engineered features:
    - promised_hours (int)
    - is_cod (int 0/1)
    - is_weekend (int 0/1)
    - dispatch_hour (int)
    - log_declared_value (natural log of declared_value_inr)
    """
    df = clean.copy()
    # Left merge on hub_id with hubs table
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    # Drop any overlapping cols from hubs if needed, but hubs has only those 5
    df = df.merge(hubs[hub_cols], on="hub_id", how="left")

    # promised_hours
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)

    # is_cod: 1 when payment_mode is COD, else 0 (Unknown gives 0)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # is_weekend: 1 when Saturday or Sunday, else 0
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)

    # dispatch_hour: hour of dispatch timestamp as integer
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)

    # log_declared_value: natural log of declared_value_inr
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
