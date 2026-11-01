"""Part B: Delivery-Hours Regression."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import src.shipment_prep as sp


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Read modelling table
    df = pd.read_csv("data/modelling_table.csv")
    X = df[sp.FEATURES]
    y = df["delivery_hours"]

    # Split into train (70%), val (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 2. Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
    test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())

    checks = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(train_scaled_max_abs_mean, 8)),
        ("test_scaled_max_abs_mean", round(test_scaled_max_abs_mean, 4)),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
        "outputs/part_b_checks.csv", index=False
    )
    print("outputs/part_b_checks.csv written.")

    # 3. Baseline model
    val_mean_pred = np.full(len(y_val), y_train.mean())
    base_mae = float(mean_absolute_error(y_val, val_mean_pred))
    base_mse = float(mean_squared_error(y_val, val_mean_pred))
    base_rmse = float(np.sqrt(base_mse))
    base_r2 = float(r2_score(y_val, val_mean_pred))

    # 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    tree_objects = {}

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_rmse = float(np.sqrt(mean_squared_error(y_train, dt.predict(X_train))))
        v_rmse = float(np.sqrt(mean_squared_error(y_val, dt.predict(X_val))))
        depth_label = "unlimited" if d is None else d
        tree_records.append({
            "max_depth": depth_label,
            "train_rmse": round(tr_rmse, 4),
            "val_rmse": round(v_rmse, 4),
            "_d": d,
            "_val_rmse": v_rmse,
        })
        tree_objects[d] = dt

    tree_depth_df = pd.DataFrame(tree_records)
    tree_depth_df[["max_depth", "train_rmse", "val_rmse"]].to_csv(
        "outputs/part_b_tree_depth.csv", index=False
    )
    print("outputs/part_b_tree_depth.csv written.")

    # Best depth: lowest val_rmse, ties -> smaller depth
    best_item = min(
        tree_records,
        key=lambda r: (r["_val_rmse"], r["_d"] if r["_d"] is not None else 9999),
    )
    best_depth = best_item["_d"]
    best_dt = tree_objects[best_depth]

    # 5. Four candidate models
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": best_dt,
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = [
        {
            "model": "mean_baseline",
            "split": "val",
            "MAE": round(base_mae, 4),
            "MSE": round(base_mse, 4),
            "RMSE": round(base_rmse, 4),
            "R2": round(base_r2, 4),
        }
    ]

    val_rmses = {}
    for name, model in candidates.items():
        if name != "decision_tree":  # best_dt was already fitted
            model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, val_preds))
        mse = float(mean_squared_error(y_val, val_preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_val, val_preds))
        val_rmses[name] = rmse
        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })

    # 6. Learning-rate experiment
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gb_lr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb_lr.fit(X_train, y_train)
        pred_lr = gb_lr.predict(X_val)
        mse_lr = float(mean_squared_error(y_val, pred_lr))
        rmse_lr = float(np.sqrt(mse_lr))
        r2_lr = float(r2_score(y_val, pred_lr))
        lr_records.append({
            "learning_rate": lr,
            "val_rmse": round(rmse_lr, 4),
            "val_r2": round(r2_lr, 4),
        })
    pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    print("outputs/part_b_gb_learning_rate.csv written.")

    # 7. Cross-validated comparison and final model
    # Rank candidates by validation RMSE
    ranked_candidates = sorted(val_rmses.items(), key=lambda x: x[1])
    top_two = [ranked_candidates[0][0], ranked_candidates[1][0]]

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []
    for model_name in top_two:
        m = candidates[model_name]
        scores = cross_val_score(m, X_train_val, y_train_val, cv=kf)
        cv_records.append({
            "model": model_name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })
    pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)
    print("outputs/part_b_cv.csv written.")

    # Final model is lowest validation RMSE candidate
    final_model_name = ranked_candidates[0][0]
    final_model = candidates[final_model_name]

    # Evaluate final model on test set once
    test_preds = final_model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, test_preds))
    test_mse = float(mean_squared_error(y_test, test_preds))
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = float(r2_score(y_test, test_preds))

    metrics_rows.append({
        "model": final_model_name,
        "split": "test",
        "MAE": round(test_mae, 4),
        "MSE": round(test_mse, 4),
        "RMSE": round(test_rmse, 4),
        "R2": round(test_r2, 4),
    })

    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
    print("outputs/part_b_metrics.csv written.")


if __name__ == "__main__":
    main()
