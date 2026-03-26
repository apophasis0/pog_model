import numpy as np
import pandas as pd

ID_COLS = ["ketto_num"]
TARGET_COLS = [
    "win_flag",
    "bt_place_flag",
    "bt_win_flag",
    "graded_win_flag",
    "positive_prize_flag",
    "pog_total_prize",
    "label_complete",
]

CATEGORICAL_COLS = [
    "sex_cd",
    "hinsyu_cd",
    "sanku_mochi_kubun",
    "breeder_code",
    "sanchi_name",
    "tozai_cd",
    "chokyosi_code",
    "banusi_code",
    "sire_hansyoku_num",
    "dam_hansyoku_num",
    "damsire_hansyoku_num",
]

NUMERIC_COLS = [
    "birth_month",
    "import_year",
    "days_birth_to_reg",
    "sire_prior_foals",
    "sire_prior_win_rate",
    "sire_prior_bt_rate",
    "sire_prior_graded_win_rate",
    "sire_prior_avg_log_prize",
    "sire_prior_med_prize",
    "dam_prior_foals",
    "dam_prior_win_rate",
    "dam_prior_bt_rate",
    "dam_prior_graded_win_rate",
    "dam_prior_avg_log_prize",
    "dam_prior_med_prize",
    "breeder_prior_foals",
    "breeder_prior_win_rate",
    "breeder_prior_bt_rate",
    "breeder_prior_graded_win_rate",
    "breeder_prior_avg_log_prize",
    "breeder_prior_med_prize",
    "trainer_prior_foals",
    "trainer_prior_win_rate",
    "trainer_prior_bt_rate",
    "trainer_prior_graded_win_rate",
    "trainer_prior_avg_log_prize",
    "trainer_prior_med_prize",
]

DYNAMIC_NUMERIC_COLS = [
    "starts_to_asof",
    "wins_to_asof",
    "best_finish_to_asof",
    "avg_odds_to_asof",
    "min_odds_to_asof",
    "avg_ninki_to_asof",
    "total_prize_to_asof",
    "total_honsyokin_to_asof",
    "last_finish",
    "last_odds",
    "last_ninki",
    "days_since_last_start",
]

def merge_dynamic(static_df: pd.DataFrame, dynamic_df: pd.DataFrame) -> pd.DataFrame:
    df = static_df.merge(dynamic_df, on="ketto_num", how="left")
    return df

def prepare_matrix(df: pd.DataFrame, use_dynamic: bool = False):
    cat_cols = CATEGORICAL_COLS.copy()
    num_cols = NUMERIC_COLS.copy()

    if use_dynamic:
        num_cols += DYNAMIC_NUMERIC_COLS

    work = df.copy()

    for c in cat_cols:
        if c in work.columns:
            work[c] = work[c].astype("string").fillna("NA")

    for c in num_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    work[num_cols] = work[num_cols].fillna(0.0)

    feature_cols = [c for c in cat_cols + num_cols if c in work.columns]

    X = work[feature_cols]
    y = {
        "win_flag": work["win_flag"].astype(int) if "win_flag" in work.columns else None,
        "bt_place_flag": work["bt_place_flag"].astype(int) if "bt_place_flag" in work.columns else None,
        "bt_win_flag": work["bt_win_flag"].astype(int) if "bt_win_flag" in work.columns else None,
        "graded_win_flag": work["graded_win_flag"].astype(int) if "graded_win_flag" in work.columns else None,
        "positive_prize_flag": work["positive_prize_flag"].astype(int) if "positive_prize_flag" in work.columns else None,
        "pog_total_prize": work["pog_total_prize"].astype(float) if "pog_total_prize" in work.columns else None,
    }
    return X, y, feature_cols, cat_cols

def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_pog_total_prize"] = np.log1p(out["pog_total_prize"].clip(lower=0))
    return out
