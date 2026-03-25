import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from dotenv import load_dotenv

from config import Config
from data import load_training_frame, load_scoring_frame, load_dynamic_features, save_predictions
from features import prepare_matrix, merge_dynamic, add_log_target
from eval import evaluate_binary, evaluate_regression, topk_summary

def split_by_birth_year(df: pd.DataFrame, cfg: Config):
    train_df = df[
        (df["birth_year"] >= cfg.train_birth_year_start) &
        (df["birth_year"] <= cfg.train_birth_year_end)
    ].copy()

    valid_df = df[
        (df["birth_year"] >= cfg.valid_birth_year_start) &
        (df["birth_year"] <= cfg.valid_birth_year_end)
    ].copy()

    test_df = df[
        (df["birth_year"] >= cfg.test_birth_year_start) &
        (df["birth_year"] <= cfg.test_birth_year_end)
    ].copy()

    return train_df, valid_df, test_df

def build_pool(X, y, cat_cols):
    return Pool(X, label=y, cat_features=[c for c in cat_cols if c in X.columns])

def train_binary_model(train_df, valid_df, target, use_dynamic=False):
    X_train, y_train, _, cat_cols = prepare_matrix(train_df, use_dynamic=use_dynamic)
    X_valid, y_valid, _, _ = prepare_matrix(valid_df, use_dynamic=use_dynamic)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=100
    )

    model.fit(
        build_pool(X_train, y_train[target], cat_cols),
        eval_set=build_pool(X_valid, y_valid[target], cat_cols),
        use_best_model=True
    )
    return model

def train_positive_regressor(train_df, valid_df, use_dynamic=False):
    train_pos = train_df[train_df["pog_total_prize"] > 0].copy()
    valid_pos = valid_df[valid_df["pog_total_prize"] > 0].copy()

    train_pos = add_log_target(train_pos)
    valid_pos = add_log_target(valid_pos)

    X_train, _, _, cat_cols = prepare_matrix(train_pos, use_dynamic=use_dynamic)
    X_valid, _, _, _ = prepare_matrix(valid_pos, use_dynamic=use_dynamic)

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=100
    )

    model.fit(
        build_pool(X_train, train_pos["log_pog_total_prize"], cat_cols),
        eval_set=build_pool(X_valid, valid_pos["log_pog_total_prize"], cat_cols),
        use_best_model=True
    )
    return model

def predict_binary(model, df, target, use_dynamic=False):
    X, y, _, _ = prepare_matrix(df, use_dynamic=use_dynamic)
    prob = model.predict_proba(X)[:, 1]
    return prob

def predict_regressor(model, df, use_dynamic=False):
    X, _, _, _ = prepare_matrix(df, use_dynamic=use_dynamic)
    pred_log = model.predict(X)
    return pred_log

def main():
    load_dotenv()
    cfg = Config()

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_training_frame(cfg)

    if cfg.use_dynamic_features:
        dyn_parts = []
        for by in sorted(df["birth_year"].unique()):
            dyn = load_dynamic_features(cfg, int(by))
            dyn_parts.append(dyn.assign(birth_year=by))
        dyn_df = pd.concat(dyn_parts, ignore_index=True)
        df = df.merge(dyn_df.drop(columns=["birth_year"]), on="ketto_num", how="left")

    train_df, valid_df, test_df = split_by_birth_year(df, cfg)

    # 1) milestone models
    win_model = train_binary_model(train_df, valid_df, "win_flag", use_dynamic=cfg.use_dynamic_features)
    bt_model = train_binary_model(train_df, valid_df, "bt_place_flag", use_dynamic=cfg.use_dynamic_features)
    graded_model = train_binary_model(train_df, valid_df, "graded_win_flag", use_dynamic=cfg.use_dynamic_features)
    pos_model = train_binary_model(train_df, valid_df, "positive_prize_flag", use_dynamic=cfg.use_dynamic_features)

    # 2) hurdle positive amount model
    prize_model = train_positive_regressor(train_df, valid_df, use_dynamic=cfg.use_dynamic_features)

    # 保存模型
    joblib.dump(win_model, "models/win_model.joblib")
    joblib.dump(bt_model, "models/bt_model.joblib")
    joblib.dump(graded_model, "models/graded_model.joblib")
    joblib.dump(pos_model, "models/pos_model.joblib")
    joblib.dump(prize_model, "models/prize_model.joblib")

    # 测试集评估
    test_pred = test_df[["ketto_num", "birth_year", "pog_total_prize", "win_flag", "bt_place_flag", "graded_win_flag", "positive_prize_flag"]].copy()
    test_pred["p_win"] = predict_binary(win_model, test_df, "win_flag", use_dynamic=cfg.use_dynamic_features)
    test_pred["p_bt_place"] = predict_binary(bt_model, test_df, "bt_place_flag", use_dynamic=cfg.use_dynamic_features)
    test_pred["p_graded_win"] = predict_binary(graded_model, test_df, "graded_win_flag", use_dynamic=cfg.use_dynamic_features)
    test_pred["p_positive_prize"] = predict_binary(pos_model, test_df, "positive_prize_flag", use_dynamic=cfg.use_dynamic_features)
    test_pred["pred_log_prize_pos"] = predict_regressor(prize_model, test_df, use_dynamic=cfg.use_dynamic_features)
    test_pred["expected_pog_prize"] = test_pred["p_positive_prize"] * (np.expm1(test_pred["pred_log_prize_pos"]))

    metrics = {}
    metrics.update(evaluate_binary(test_pred["win_flag"], test_pred["p_win"], "win"))
    metrics.update(evaluate_binary(test_pred["bt_place_flag"], test_pred["p_bt_place"], "bt_place"))
    metrics.update(evaluate_binary(test_pred["graded_win_flag"], test_pred["p_graded_win"], "graded_win"))
    metrics.update(evaluate_binary(test_pred["positive_prize_flag"], test_pred["p_positive_prize"], "positive_prize"))
    metrics.update(evaluate_regression(test_pred["pog_total_prize"], test_pred["pred_log_prize_pos"], "prize_pos"))

    print("=== TEST METRICS ===")
    for k, v in metrics.items():
        print(k, v)

    print("=== TOP50 SUMMARY ===")
    print(topk_summary(test_pred, "expected_pog_prize", k=50))

    test_pred.to_csv("outputs/test_predictions.csv", index=False)

    # 当前 cohort 打分
    score_df = load_scoring_frame(cfg)
    if cfg.use_dynamic_features:
        score_dyn = load_dynamic_features(cfg, cfg.target_birth_year)
        score_df = merge_dynamic(score_df, score_dyn)

    pred_df = score_df[["ketto_num", "birth_year"]].copy()
    pred_df["p_win"] = predict_binary(win_model, score_df, "win_flag", use_dynamic=cfg.use_dynamic_features)
    pred_df["p_bt_place"] = predict_binary(bt_model, score_df, "bt_place_flag", use_dynamic=cfg.use_dynamic_features)
    pred_df["p_graded_win"] = predict_binary(graded_model, score_df, "graded_win_flag", use_dynamic=cfg.use_dynamic_features)
    pred_df["p_positive_prize"] = predict_binary(pos_model, score_df, "positive_prize_flag", use_dynamic=cfg.use_dynamic_features)
    pred_df["pred_log_prize_pos"] = predict_regressor(prize_model, score_df, use_dynamic=cfg.use_dynamic_features)
    pred_df["expected_pog_prize"] = pred_df["p_positive_prize"] * (np.expm1(pred_df["pred_log_prize_pos"]))

    pred_df["model_name"] = cfg.model_name
    pred_df["model_version"] = cfg.model_version
    pred_df["asof_date"] = pd.to_datetime(cfg.asof_date)
    pred_df = pred_df[
        [
            "model_name", "model_version", "asof_date", "birth_year", "ketto_num",
            "p_win", "p_bt_place", "p_graded_win", "p_positive_prize",
            "expected_pog_prize", "pred_log_prize_pos"
        ]
    ]
    pred_df.to_csv("outputs/current_cohort_predictions.csv", index=False)
    save_predictions(cfg, pred_df)

if __name__ == "__main__":
    main()
