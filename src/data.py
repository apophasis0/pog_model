import pandas as pd
from sqlalchemy import create_engine, text
from config import Config
from typing import Protocol

class HasDbUrl(Protocol):
    """Protocol for config objects that have db_url attribute."""
    db_url: str
    train_birth_year_start: int
    train_birth_year_end: int
    test_birth_year_start: int
    test_birth_year_end: int
    target_birth_year: int
    asof_date: any

STATIC_FEATURE_VIEW = "pog.mv_static_features_v2"


def get_engine(cfg: HasDbUrl):
    return create_engine(cfg.db_url)


def load_completed_birth_years(cfg: HasDbUrl):
    sql = text("""
    select distinct birth_year
    from pog.mv_horse_labels
    where label_complete = true
    order by birth_year
    """)
    engine = get_engine(cfg)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    return df["birth_year"].astype(int).tolist()


def load_training_frame(cfg: HasDbUrl) -> pd.DataFrame:
    sql = text(f"""
    select
        f.*,
        l.win_flag,
        l.bt_place_flag,
        l.bt_win_flag,
        l.graded_win_flag,
        l.positive_prize_flag,
        l.pog_total_prize,
        l.pog_total_prize_ge_10m_flag,
        l.pog_total_prize_ge_30m_flag,
        l.label_complete
    from {STATIC_FEATURE_VIEW} f
    join pog.mv_horse_labels l
      on f.ketto_num = l.ketto_num
    where f.birth_year between :train_start and :test_end
      and l.label_complete = true
      and f.is_jra_registered = true
    """)
    engine = get_engine(cfg)
    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "train_start": cfg.train_birth_year_start,
                "test_end": cfg.test_birth_year_end,
            },
        )
    return df


def load_all_labeled_frame(cfg: HasDbUrl) -> pd.DataFrame:
    """Load all label-complete JRA-registered horses (for rolling backtest)."""
    sql = text(f"""
    select
        f.*,
        l.win_flag,
        l.bt_place_flag,
        l.bt_win_flag,
        l.graded_win_flag,
        l.positive_prize_flag,
        l.pog_total_prize,
        l.pog_total_prize_ge_10m_flag,
        l.pog_total_prize_ge_30m_flag,
        l.label_complete
    from {STATIC_FEATURE_VIEW} f
    join pog.mv_horse_labels l
      on f.ketto_num = l.ketto_num
    where l.label_complete = true
      and f.is_jra_registered = true
    """)
    engine = get_engine(cfg)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    return df


def load_scoring_frame(cfg: HasDbUrl) -> pd.DataFrame:
    sql = text(f"""
    select *
    from {STATIC_FEATURE_VIEW}
    where birth_year = :target_birth_year
      and is_jra_registered = true
    """)
    engine = get_engine(cfg)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"target_birth_year": cfg.target_birth_year})
    return df


def load_dynamic_features(cfg: HasDbUrl, birth_year: int) -> pd.DataFrame:
    sql = text("""
    select *
    from pog.fn_dynamic_features(:birth_year, :asof_date)
    """)
    engine = get_engine(cfg)
    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "birth_year": birth_year,
                "asof_date": cfg.asof_date,
            },
        )
    return df


def save_predictions(cfg: HasDbUrl, pred_df: pd.DataFrame):
    engine = get_engine(cfg)
    pred_df.to_sql(
        "model_predictions",
        engine,
        schema="pog",
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
