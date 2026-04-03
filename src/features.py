from dataclasses import dataclass, fields
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
    "pog_total_prize_ge_10m_flag",
    "pog_total_prize_ge_30m_flag",
    "label_complete",
]

# =========
# Categorical features
# =========
BASE_CATEGORICAL_COLS = [
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

NEW_CATEGORICAL_COLS = [
    "granddam_hansyoku_num",
]


# =========
# Numeric features
# =========
BASE_NUMERIC_COLS = [
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

# -- New numeric feature groups (for ablation) --
NEW_NUMERIC_DAM_SIRE_AGE = [
    "dam_age_at_foaling",
    "sire_age_at_foaling",
]

NEW_NUMERIC_DAMSIRE_STATS = [
    "damsire_prior_foals",
    "damsire_prior_win_rate",
    "damsire_prior_bt_place_rate",
    "damsire_prior_bt_win_rate",
    "damsire_prior_graded_win_rate",
    "damsire_prior_avg_log_prize",
    "damsire_prior_med_prize",
    "damsire_prior_best_prize",
]

NEW_NUMERIC_MATERNAL_SIB = [
    "prior_maternal_sib_count",
    "prior_maternal_sib_win_count",
    "prior_maternal_sib_bt_place_count",
    "prior_maternal_sib_bt_win_count",
    "prior_maternal_sib_graded_win_count",
    "prior_maternal_sib_win_rate",
    "prior_maternal_sib_bt_place_rate",
    "prior_maternal_sib_bt_win_rate",
    "prior_maternal_sib_graded_win_rate",
    "prior_maternal_sib_avg_log_prize",
    "prior_maternal_sib_med_prize",
    "prior_maternal_sib_best_prize",
]

NEW_NUMERIC_FULL_SIB = [
    "prior_full_sib_count",
    "prior_full_sib_win_rate",
    "prior_full_sib_bt_place_rate",
    "prior_full_sib_bt_win_rate",
    "prior_full_sib_graded_win_rate",
    "prior_full_sib_avg_log_prize",
    "prior_full_sib_med_prize",
    "prior_full_sib_best_prize",
]

NEW_NUMERIC_NICK_STATS = [
    "nick_prior_foals",
    "nick_prior_win_rate",
    "nick_prior_bt_place_rate",
    "nick_prior_bt_win_rate",
    "nick_prior_graded_win_rate",
    "nick_prior_avg_log_prize",
    "nick_prior_best_prize",
]

NEW_NUMERIC_BREEDER_TRAINER = [
    "breeder_trainer_prior_foals",
    "breeder_trainer_prior_win_rate",
    "breeder_trainer_prior_bt_place_rate",
    "breeder_trainer_prior_bt_win_rate",
    "breeder_trainer_prior_graded_win_rate",
    "breeder_trainer_prior_avg_log_prize",
    "breeder_trainer_prior_best_prize",
]

# Combined list (kept for backward compat references)
NEW_NUMERIC_COLS = (
    NEW_NUMERIC_DAM_SIRE_AGE
    + NEW_NUMERIC_DAMSIRE_STATS
    + NEW_NUMERIC_MATERNAL_SIB
    + NEW_NUMERIC_FULL_SIB
    + NEW_NUMERIC_NICK_STATS
    + NEW_NUMERIC_BREEDER_TRAINER
)

NUMERIC_COLS = BASE_NUMERIC_COLS + NEW_NUMERIC_COLS
CATEGORICAL_COLS = BASE_CATEGORICAL_COLS + NEW_CATEGORICAL_COLS


# =========
# FeatureSet — toggleable feature groups for ablation
# =========

@dataclass
class FeatureSet:
    """Controls which feature groups are active for training/prediction."""
    base_categorical: bool = True
    new_categorical: bool = True
    base_numeric: bool = True
    new_numeric_dam_sire_age: bool = True
    new_numeric_damsire_stats: bool = True
    new_numeric_maternal_sib: bool = True
    new_numeric_full_sib: bool = True
    new_numeric_nick_stats: bool = True
    new_numeric_breeder_trainer: bool = True

    def get_categorical_cols(self) -> list[str]:
        cols: list[str] = []
        if self.base_categorical:
            cols += BASE_CATEGORICAL_COLS
        if self.new_categorical:
            cols += NEW_CATEGORICAL_COLS
        return cols

    def get_numeric_cols(self) -> list[str]:
        cols: list[str] = []
        if self.base_numeric:
            cols += BASE_NUMERIC_COLS
        if self.new_numeric_dam_sire_age:
            cols += NEW_NUMERIC_DAM_SIRE_AGE
        if self.new_numeric_damsire_stats:
            cols += NEW_NUMERIC_DAMSIRE_STATS
        if self.new_numeric_maternal_sib:
            cols += NEW_NUMERIC_MATERNAL_SIB
        if self.new_numeric_full_sib:
            cols += NEW_NUMERIC_FULL_SIB
        if self.new_numeric_nick_stats:
            cols += NEW_NUMERIC_NICK_STATS
        if self.new_numeric_breeder_trainer:
            cols += NEW_NUMERIC_BREEDER_TRAINER
        return cols

    def describe(self) -> str:
        """Human-readable summary of enabled feature groups."""
        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            if val:
                parts.append(f"+{f.name}")
            else:
                parts.append(f"-{f.name}")
        return ", ".join(parts)

    def short_name(self) -> str:
        """Short name listing only the disabled groups (for labeling experiments)."""
        disabled = [f.name for f in fields(self) if not getattr(self, f.name)]
        if not disabled:
            return "all_features"
        return "no_" + "_".join(d.replace("new_numeric_", "").replace("new_categorical", "new_cat") for d in disabled)


# =========
# High-cardinality categorical handling
# =========

def fit_category_frequencies(
    df: pd.DataFrame, cat_cols: list[str]
) -> dict[str, dict[str, int]]:
    """Count the frequency of each category value in the training data.

    Args:
        df: Training DataFrame.
        cat_cols: List of categorical column names.

    Returns:
        A dict mapping column name -> {category_value: count}.
    """
    freq_maps: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        if col in df.columns:
            freq_maps[col] = df[col].astype("string").fillna("NA").value_counts().to_dict()
    return freq_maps


def apply_rare_filter(
    df: pd.DataFrame,
    cat_cols: list[str],
    freq_maps: dict[str, dict[str, int]],
    min_count: int = 3,
) -> pd.DataFrame:
    """Replace category values appearing fewer than *min_count* times with 'RARE'.

    Args:
        df: DataFrame to filter.
        cat_cols: List of categorical column names.
        freq_maps: Frequency maps from `fit_category_frequencies`.
        min_count: Minimum occurrence threshold.

    Returns:
        DataFrame with rare categories replaced by "RARE".
    """
    work = df.copy()
    for col in cat_cols:
        if col in freq_maps and col in work.columns:
            fmap = freq_maps[col]
            work[col] = work[col].astype("string").fillna("NA").apply(
                lambda x: x if fmap.get(x, 0) >= min_count else "RARE"
            )
    return work


def prepare_matrix(df: pd.DataFrame, feature_set: FeatureSet | None = None):
    """Build feature matrix X and target dict y from a DataFrame.

    Args:
        df: Input DataFrame with feature and target columns.
        feature_set: Controls which feature groups are active.
                     Defaults to FeatureSet() (all features ON).
    """
    if feature_set is None:
        feature_set = FeatureSet()

    work = df.copy()

    cat_cols = feature_set.get_categorical_cols()
    num_cols = feature_set.get_numeric_cols()

    existing_cat_cols = [c for c in cat_cols if c in work.columns]
    existing_num_cols = [c for c in num_cols if c in work.columns]

    # categorical
    for c in existing_cat_cols:
        work[c] = work[c].astype("string").fillna("NA")

    # numeric
    for c in existing_num_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # NOTE: Intentionally NOT filling NaN with 0.0 here.
    # CatBoost natively handles NaN values via its Min/Max Split strategy,
    # learning the semantic difference between "missing" (e.g., first-year sire)
    # and "zero" (e.g., sire with 100 progeny but no wins).
    # Filling NaN with 0.0 would confuse these two distinct cases.

    feature_cols = existing_cat_cols + existing_num_cols
    X = work[feature_cols]

    y = {}
    for tc in TARGET_COLS:
        if tc in work.columns:
            if tc == "pog_total_prize":
                y[tc] = pd.to_numeric(work[tc], errors="coerce").fillna(0.0).astype(float)
            elif tc == "label_complete":
                y[tc] = work[tc]
            else:
                y[tc] = pd.to_numeric(work[tc], errors="coerce").fillna(0).astype(int)

    return X, y, feature_cols, existing_cat_cols


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_pog_total_prize"] = np.log1p(
        pd.to_numeric(out["pog_total_prize"], errors="coerce").fillna(0.0).clip(lower=0)
    )
    return out
