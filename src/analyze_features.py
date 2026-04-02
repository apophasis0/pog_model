"""
analyze_features.py - Compute global feature importances and SHAP values for top components.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set Japanese font for matplotlib to prevent gibberish
plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'sans-serif']

import shap
from catboost import Pool
from dotenv import load_dotenv

from config import Config
from features import prepare_matrix
from pipeline import build_pool
from score_cohorts import load_bundle, load_cohort_frame

def map_weight_to_model_names(feat: str) -> list[str]:
    mapping = {
        "r_p_prize_ge_10m": ["prize_ge_10m_model"],
        "r_p_prize_ge_30m": ["prize_ge_30m_model"],
        "r_p_graded_win": ["graded_given_bt_win_model"],
        "r_q90_prize": ["q90_model"],
        "r_expected_pog_prize": ["positive_prize_model", "prize_model"],
    }
    return mapping.get(feat, [])

def extract_feature_importances(model, X, cat_cols, top_n=20):
    pool = build_pool(X, None, cat_cols)
    # Note: PredictionValuesChange is standard feature importance in Catboost
    try:
        importances = model.get_feature_importance(pool, type="PredictionValuesChange")
    except Exception as e:
        print(f"[WARN] Failed to get PredictionValuesChange using pool: {e}. Trying without pool.")
        importances = model.get_feature_importance(type="PredictionValuesChange")
        
    df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df

def generate_local_shap_plot(model, model_name, X_row, cat_cols, out_dir, ketto_num):
    print(f"Calculating local SHAP for {model_name} (ketto_num={ketto_num})...")
    pool = build_pool(X_row, None, cat_cols)
    shap_vals_matrix = model.get_feature_importance(pool, type="ShapValues")
    
    shap_vals = shap_vals_matrix[0, :-1]
    base_value = shap_vals_matrix[0, -1]
    
    exp = shap.Explanation(
        values=shap_vals, 
        base_values=base_value, 
        data=X_row.iloc[0].values, 
        feature_names=list(X_row.columns)
    )
    
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(exp, show=False)
    plt.title(f"Local SHAP (Waterfall): {model_name} [ketto_num={ketto_num}]")
    out_path = os.path.join(out_dir, f"{model_name}_{ketto_num}_waterfall.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved local SHAP plot to {out_path}")

def generate_shap_plots(model, model_name, X, cat_cols, out_dir):
    print(f"Calculating SHAP for {model_name}...")
    pool = build_pool(X, None, cat_cols)
    
    # SHAP values using CatBoost internal implementation (recommended for categorical features)
    shap_vals_matrix = model.get_feature_importance(pool, type="ShapValues")
    
    # shap_vals_matrix is (n_samples, n_features + 1)
    shap_values = shap_vals_matrix[:, :-1]
    
    plt.figure(figsize=(12, 10))
    # Render standard SHAP summary plot (dot)
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary (Dot): {model_name}")
    out_path = os.path.join(out_dir, f"{model_name}_shap_summary.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"  -> Saved SHAP plot to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Feature importance & SHAP analysis")
    parser.add_argument("--model-dir", default="models_recommended", help="Input model bundle directory")
    parser.add_argument("--years", nargs="+", type=int, default=[2023], help="Cohort year(s) to analyze for SHAP")
    parser.add_argument("--output-dir", default="outputs/feature_analysis", help="Output directory for plots/csvs")
    parser.add_argument("--top-ceiling", type=int, default=3, help="Top N ceiling weights whose sub-models to explain")
    parser.add_argument("--ketto-num", type=str, default=None, help="Specific horse ID to generate local SHAP plots for")
    args = parser.parse_args()

    cfg = Config()
    
    csv_dir = os.path.join(args.output_dir, "csvs")
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"=== LOADING BUNDLE FROM {args.model_dir} ===")
    bundle = load_bundle(args.model_dir)
    
    weights = bundle.ceiling_weights
    print("Ceiling Weights in bundle:")
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")
        
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top_w = sorted_weights[:args.top_ceiling]
    print(f"\nTop {args.top_ceiling} features in score_ceiling:")
    
    target_model_names = set()
    for feat, w in top_w:
        print(f"  {feat}: {w:.4f}")
        for m_name in map_weight_to_model_names(feat):
            target_model_names.add(m_name)
            
    print(f"\nModels selected for SHAP analysis: {list(target_model_names)}")
    
    print("\n=== LOADING DATA FOR COHORTS: {args.years} ===")
    dfs = []
    for yr in args.years:
        df_yr = load_cohort_frame(cfg, yr)
        if not df_yr.empty:
            dfs.append(df_yr)
            
    if not dfs:
        print("[ERROR] No data found for specified years. Exiting.")
        return
        
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total shape for SHAP and Feature Importance analysis: {df_all.shape}")
    
    # Prepare X matrix ensuring it matches training time (dummy target y will be ignored)
    X, _, _, cat_cols = prepare_matrix(df_all, feature_set=bundle.feature_set)
    
    if args.ketto_num:
        if args.ketto_num not in df_all["ketto_num"].astype(str).values:
            print(f"[ERROR] ketto_num {args.ketto_num} not found in the loaded data.")
            return
            
        idx = df_all.index[df_all["ketto_num"].astype(str) == str(args.ketto_num)][0]
        X_row = X.iloc[[idx]]
        print(f"Found {args.ketto_num} at index {idx}.")
        
        for model_name in target_model_names:
            model = getattr(bundle, model_name)
            if model is None: continue
            generate_local_shap_plot(model, model_name, X_row, cat_cols, plot_dir, args.ketto_num)
        
        print("\n=== Local Plot DONE ===")
        print(f"Output directory: {plot_dir}")
        return

    for model_name in target_model_names:
        model = getattr(bundle, model_name)
        if model is None:
            print(f"[WARN] Attribute '{model_name}' is None, skipping.")
            continue
            
        print(f"\n--- Processing Model: {model_name} ---")
        # Global Feature Importance
        df_imp = extract_feature_importances(model, X, cat_cols)
        out_csv = os.path.join(csv_dir, f"{model_name}_importance.csv")
        df_imp.to_csv(out_csv, index=False)
        print(f"  -> Saved global feature importance CSV to: {out_csv}")
        print(f"     Top 3 features: {df_imp['feature'].head(3).tolist()}")
        
        # SHAP calculation
        generate_shap_plots(model, model_name, X, cat_cols, plot_dir)
        
    print(f"\n=== DONE ===")
    print(f"Output directories:\n  CSV: {csv_dir}\n  Plots: {plot_dir}")
    
if __name__ == "__main__":
    load_dotenv()
    main()
