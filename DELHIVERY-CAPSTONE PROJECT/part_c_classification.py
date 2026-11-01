"""Part C: Delay Classifier and Alert Threshold."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import src.shipment_prep as sp


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Stratified split
    df = pd.read_csv("data/modelling_table.csv")
    X = df[sp.FEATURES]
    y = df["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 2. Decision tree tuning over depths: 2, 3, 4, 5, 6, 8, 10
    dt_records = []
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        pred = dt.predict(X_val)
        val_f1 = float(f1_score(y_val, pred, zero_division=0))
        dt_records.append((d, val_f1, dt))

    best_dt_d, best_dt_f1, best_dt = max(dt_records, key=lambda x: (x[1], -x[0]))

    # Blind baseline: predict 0 for all validation rows
    blind_pred = np.zeros(len(y_val), dtype=int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_val, blind_pred, labels=[0, 1]).ravel()
    blind_acc = float(accuracy_score(y_val, blind_pred))
    blind_row = {
        "model": "blind_all_on_time",
        "TN": int(tn_b),
        "FP": int(fp_b),
        "FN": int(fn_b),
        "TP": int(tp_b),
        "accuracy": round(blind_acc, 4),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    # Logistic Regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_val)
    tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_val, lr_pred, labels=[0, 1]).ravel()
    lr_row = {
        "model": "logistic_regression",
        "TN": int(tn_lr),
        "FP": int(fp_lr),
        "FN": int(fn_lr),
        "TP": int(tp_lr),
        "accuracy": round(float(accuracy_score(y_val, lr_pred)), 4),
        "precision": round(float(precision_score(y_val, lr_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, lr_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_val, lr_pred, zero_division=0)), 4),
    }

    # Best Decision Tree
    dt_pred = best_dt.predict(X_val)
    tn_dt, fp_dt, fn_dt, tp_dt = confusion_matrix(y_val, dt_pred, labels=[0, 1]).ravel()
    dt_row = {
        "model": f"decision_tree_depth_{best_dt_d}",
        "TN": int(tn_dt),
        "FP": int(fp_dt),
        "FN": int(fn_dt),
        "TP": int(tp_dt),
        "accuracy": round(float(accuracy_score(y_val, dt_pred)), 4),
        "precision": round(float(precision_score(y_val, dt_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, dt_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_val, dt_pred, zero_division=0)), 4),
    }

    # Default Random Forest
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    rf_pred = rf_default.predict(X_val)
    tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_val, rf_pred, labels=[0, 1]).ravel()
    rf_row = {
        "model": "random_forest_default",
        "TN": int(tn_rf),
        "FP": int(fp_rf),
        "FN": int(fn_rf),
        "TP": int(tp_rf),
        "accuracy": round(float(accuracy_score(y_val, rf_pred)), 4),
        "precision": round(float(precision_score(y_val, rf_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, rf_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_val, rf_pred, zero_division=0)), 4),
    }

    models_df = pd.DataFrame([blind_row, lr_row, dt_row, rf_row])
    models_df.to_csv("outputs/part_c_models.csv", index=False)
    print("outputs/part_c_models.csv written.")

    # 3. Feature importance & F-score signals
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_train, y_train)
    importances = rf_all.feature_importances_

    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_

    feature_scores_df = pd.DataFrame({
        "feature": sp.FEATURES,
        "f_score": [round(float(s), 4) for s in f_scores],
        "rf_importance": [round(float(i), 4) for i in importances],
        "is_noise": [c in sp.NOISE_COLUMNS for c in sp.FEATURES],
    })
    feature_scores_df["f_rank"] = (
        feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    )
    feature_scores_df = feature_scores_df[[
        "feature", "f_score", "f_rank", "rf_importance", "is_noise"
    ]].sort_values("f_rank")
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    print("outputs/part_c_feature_scores.csv written.")

    # SelectK validation F1
    selectk_records = []
    for k in [5, 7, 10, 12, 17]:
        k_pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        k_pipe.fit(X_train, y_train)
        k_pred = k_pipe.predict(X_val)
        val_f1 = float(f1_score(y_val, k_pred, zero_division=0))
        selectk_records.append({"k": k, "val_f1": round(val_f1, 4)})
    pd.DataFrame(selectk_records).to_csv("outputs/part_c_selectk.csv", index=False)
    print("outputs/part_c_selectk.csv written.")

    # 4. Tuning with k=7 features
    selector_7 = SelectKBest(f_classif, k=7)
    X_train_7 = selector_7.fit_transform(X_train, y_train)
    X_val_7 = selector_7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    # GridSearchCV: 27 candidates, 5 folds = 135 fits
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
    )
    gs.fit(X_train_7, y_train)
    gs_pred = gs.predict(X_val_7)
    gs_val_f1 = float(f1_score(y_val, gs_pred, zero_division=0))

    # RandomizedSearchCV: 9 candidates, 5 folds = 45 fits
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
    )
    rs.fit(X_train_7, y_train)
    rs_pred = rs.predict(X_val_7)
    rs_val_f1 = float(f1_score(y_val, rs_pred, zero_division=0))

    # Untuned default forest on same 7 columns
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train_7, y_train)
    rf_def_pred = rf_def.predict(X_val_7)
    rf_def_val_f1 = float(f1_score(y_val, rf_def_pred, zero_division=0))

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(gs.best_params_),
            "best_cv_accuracy": round(float(gs.best_score_), 4),
            "val_f1": round(gs_val_f1, 4),
            "_obj": gs.best_estimator_,
            "_params": gs.best_params_,
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rs.best_params_),
            "best_cv_accuracy": round(float(rs.best_score_), 4),
            "val_f1": round(rs_val_f1, 4),
            "_obj": rs.best_estimator_,
            "_params": rs.best_params_,
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": round(rf_def_val_f1, 4),
            "_obj": rf_def,
            "_params": {"n_estimators": 100, "max_depth": "unlimited", "min_samples_leaf": 1},
        },
    ]

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df[[
        "search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"
    ]].to_csv("outputs/part_c_tuning.csv", index=False)
    print("outputs/part_c_tuning.csv written.")

    # 5. Choosing final configuration: highest val_f1
    # Ties -> default_forest_same_7_features (priority 0), grid_search (1), randomized_search (2)
    tie_priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
    sorted_tuning = sorted(
        tuning_rows,
        key=lambda r: (r["val_f1"], -tie_priority[r["search"]]),
        reverse=True,
    )
    winning_config = sorted_tuning[0]
    chosen_search_name = winning_config["search"]
    chosen_rf = winning_config["_obj"]
    chosen_params = winning_config["_params"]

    # Build final Pipeline
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", chosen_rf),
    ])
    final_pipeline.fit(X_train, y_train)

    # 6. Alert threshold
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_records = []
    for t in threshold_list:
        pred_t = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred_t, labels=[0, 1]).ravel()
        acc = float(accuracy_score(y_val, pred_t))
        prec = float(precision_score(y_val, pred_t, zero_division=0))
        rec = float(recall_score(y_val, pred_t, zero_division=0))
        f1_val = float(f1_score(y_val, pred_t, zero_division=0))
        thresh_records.append({
            "threshold": t,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_val, 4),
        })

    thresh_df = pd.DataFrame(thresh_records)
    thresh_df.to_csv("outputs/part_c_thresholds.csv", index=False)
    print("outputs/part_c_thresholds.csv written.")

    # Operations rule: highest t whose validation recall is >= 0.80 (if none reaches 0.80, use 0.1)
    qualifying = thresh_df[thresh_df["recall"] >= 0.80]
    if len(qualifying) > 0:
        chosen_threshold = float(qualifying["threshold"].max())
    else:
        chosen_threshold = 0.1

    # 7. One test evaluation
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_pred_t = (test_probs >= chosen_threshold).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, test_pred_t, labels=[0, 1]).ravel()
    test_acc = float(accuracy_score(y_test, test_pred_t))
    test_prec = float(precision_score(y_test, test_pred_t, zero_division=0))
    test_rec = float(recall_score(y_test, test_pred_t, zero_division=0))
    test_f1 = float(f1_score(y_test, test_pred_t, zero_division=0))

    final_test_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        "TN": int(tn_t),
        "FP": int(fp_t),
        "FN": int(fn_t),
        "TP": int(tp_t),
        "accuracy": round(test_acc, 4),
        "precision": round(test_prec, 4),
        "recall": round(test_rec, 4),
        "f1": round(test_f1, 4),
    }])
    final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)
    print("outputs/part_c_final_test.csv written.")

    # Final config file
    n_est = chosen_params.get("n_estimators", 100)
    max_d = chosen_params.get("max_depth", "unlimited")
    if max_d is None:
        max_d = "unlimited"
    min_leaf = chosen_params.get("min_samples_leaf", 1)

    final_config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": n_est,
        "max_depth": max_d,
        "min_samples_leaf": min_leaf,
        "threshold": chosen_threshold,
        "chosen_from": chosen_search_name,
    }])
    final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)
    print("outputs/part_c_final_config.csv written.")


if __name__ == "__main__":
    main()
