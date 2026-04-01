CREATE SCHEMA IF NOT EXISTS pog AUTHORIZATION jvadmin;

DROP TABLE IF EXISTS pog.cohort_calendar CASCADE;
DELETE FROM pog.cohort_calendar WHERE true;
create table if not exists pog.cohort_calendar
(
    id serial primary key,
    birth_year     integer not null,
    season_year    integer not null,
    rule_comment   text not null default '',
    draft_date     date    not null,
    pog_start_date date    not null,
    pog_end_date   date    not null,
    label_complete boolean not null default false
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cohort_calendar_birth_year ON pog.cohort_calendar (birth_year, rule_comment);

insert into pog.cohort_calendar (
    birth_year, season_year, rule_comment, draft_date, pog_start_date, pog_end_date, label_complete
)
select
    y as birth_year,
    y + 2 as season_year,
    'arima' AS rule_comment,
    make_date(y + 2, 6, 1) as draft_date,
    make_date(y + 2, 6, 1) as pog_start_date,
    make_date(y + 3, 12, 31) as pog_end_date,
    (current_date > make_date(y + 3, 12, 31)) as label_complete
from generate_series(1995, extract(year from current_date)::int) as gs(y)
on conflict (birth_year, rule_comment) do update
set season_year    = excluded.season_year,
    draft_date     = excluded.draft_date,
    pog_start_date = excluded.pog_start_date,
    pog_end_date   = excluded.pog_end_date,
    label_complete = excluded.label_complete;

drop materialized view if exists pog.mv_horse_master;

create materialized view pog.mv_horse_master as
with base as (select s.ketto_num,
                     (s.birth_date at time zone 'Asia/Tokyo')::date                  as birth_date,
                     extract(year from s.birth_date at time zone 'Asia/Tokyo')::int  as birth_year,
                     extract(month from s.birth_date at time zone 'Asia/Tokyo')::int as birth_month,
                     s.sex_cd,
                     s.hinsyu_cd,
                     s.sanku_mochi_kubun,
                     s.import_year,
                     s.breeder_code,
                     s.sanchi_name,
                     coalesce(u.reg_date, null)::date                                as reg_date,
                     CASE
                         WHEN u.ketto_num is not null AND
                              reg_date > '1970-01-01 00:00:00 Asia/Tokyo'::timestamp with time zone
                             THEN true
                         ELSE false END                                              AS is_jra_registered,
                     u.tozai_cd,
                     u.chokyosi_code,
                     u.chokyosi_ryakusyo,
                     u.banusi_code,
                     u.banusi_name,
                     coalesce(u.ketto_info_hansyoku_nums, s.hansyoku_num)            as pedigree_nums
              from public.sankus s
                       left join public.umas u
                                 on s.ketto_num = u.ketto_num)
select b.*,
       b.pedigree_nums[1] as sire_hansyoku_num,
       b.pedigree_nums[2] as dam_hansyoku_num,
       b.pedigree_nums[5] as damsire_hansyoku_num,
       hs.bamei           as sire_name,
       hd.bamei           as dam_name,
       hds.bamei          as damsire_name,
       case
           when b.reg_date is not null AND
                reg_date > '1970-01-01 00:00:00 Asia/Tokyo'::timestamp with time zone then (b.reg_date - b.birth_date)
           else null
           end            as days_birth_to_reg
from base b
         left join public.hansyokus hs
                   on b.pedigree_nums[1] = hs.hansyoku_num
         left join public.hansyokus hd
                   on b.pedigree_nums[2] = hd.hansyoku_num
         left join public.hansyokus hds
                   on b.pedigree_nums[5] = hds.hansyoku_num;

create unique index if not exists idx_mv_horse_master_ketto_num
    on pog.mv_horse_master (ketto_num);

create index if not exists idx_mv_horse_master_birth_year
    on pog.mv_horse_master (birth_year);

create index if not exists idx_mv_horse_master_sire
    on pog.mv_horse_master (sire_hansyoku_num);

create index if not exists idx_mv_horse_master_dam
    on pog.mv_horse_master (dam_hansyoku_num);

create index if not exists idx_mv_horse_master_breeder
    on pog.mv_horse_master (breeder_code);

create index if not exists idx_mv_horse_master_trainer
    on pog.mv_horse_master (chokyosi_code);

drop materialized view if exists pog.mv_race_result_enriched;

create materialized view pog.mv_race_result_enriched as
select ru.ketto_num,
       ru.bamei,
       (ru.race_date at time zone 'Asia/Tokyo')::date as race_date,
       ru.jyo_cd,
       ru.kaiji,
       ru.nichiji,
       ru.race_num,
       ru.barei,
       ru.kakutei_jyuni,
       ru.nyusen_jyuni,
       ru.odds,
       ru.ninki,
       ru.honsyokin,
       ru.fukasyokin,
       ru.time_diff,
       ru.soha_time,
       ru.haron_time_l3,
       ru.haron_time_l4,
       ru.jyuni1c,
       ru.jyuni2c,
       ru.jyuni3c,
       ru.jyuni4c,
       ru.kyakusitu_kubun,
       ru.i_jyo_cd,
       rd.grade_cd,
       rd.syubetu_cd,
       rd.track_cd,
       rd.kubun,
       rd.jyoken_name,
       rd.kyori,
       rd.siba_baba_cd,
       rd.dirt_baba_cd,
       rd.syusso_tosu,

       case
           when substr(track_cd::text, 4) IN ('turf', 'dirt') then true
           else false
           end                                        as is_flat,

       case
           when rd.grade_cd in ('g1', 'g2', 'g3') then true
           else false
           end                                        as is_graded,

       case
           when rd.grade_cd = 'listed' then true
           else false
           end                                        as is_listed,

       case
           when rd.grade_cd in ('g1', 'g2', 'g3', 'listed') then true
           else false
           end                                        as is_black_type,

       CASE
           WHEN syubetu_cd = 'two_year_old' OR syubetu_cd = 'three_year_old' OR syubetu_cd = 'three_year_old_up'
               THEN true
           ELSE false
           END                                        AS is_pog_target

from public.race_umas ru
         join public.race_details rd
              on ru.race_date = rd.race_date
                  and ru.jyo_cd = rd.jyo_cd
                  and ru.kaiji = rd.kaiji
                  and ru.nichiji = rd.nichiji
                  and ru.race_num = rd.race_num;

create index if not exists idx_mv_race_result_enriched_ketto_date
    on pog.mv_race_result_enriched (ketto_num, race_date);

create index if not exists idx_mv_race_result_enriched_date
    on pog.mv_race_result_enriched (race_date);

create index if not exists idx_mv_race_result_enriched_flags
    on pog.mv_race_result_enriched (is_flat, is_black_type, is_graded);

drop materialized view if exists pog.mv_horse_labels CASCADE;

create materialized view pog.mv_horse_labels as
with horse_window as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.is_jra_registered,
        hm.reg_date,
        cc.pog_start_date,
        cc.pog_end_date,
        cc.rule_comment,
        cc.label_complete
    from pog.mv_horse_master hm
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
),
target_races as (
    select *
    from pog.mv_race_result_enriched
    where is_pog_target = true
)
select
    hw.ketto_num,
    hw.birth_year,
    hw.is_jra_registered,
    hw.reg_date,
    hw.pog_start_date,
    hw.pog_end_date,
    hw.rule_comment,
    hw.label_complete,

    count(tr.ketto_num) as starts_pog,
    sum(case when tr.kakutei_jyuni = 1 then 1 else 0 end) as wins_pog,

    max(case when tr.kakutei_jyuni = 1 then 1 else 0 end) as win_flag,

    max(case when tr.is_black_type = true and tr.kakutei_jyuni <= 3 then 1 else 0 end) as bt_place_flag,

    max(case when tr.is_black_type = true and tr.kakutei_jyuni = 1 then 1 else 0 end) as bt_win_flag,

    max(case when tr.is_graded = true and tr.kakutei_jyuni = 1 then 1 else 0 end) as graded_win_flag,

    sum(coalesce(tr.honsyokin, 0))::bigint as pog_honsyokin,
    sum(coalesce(tr.honsyokin, 0) + coalesce(tr.fukasyokin, 0))::bigint as pog_total_prize,

    max(case when coalesce(tr.honsyokin, 0) + coalesce(tr.fukasyokin, 0) > 0 then 1 else 0 end) as positive_prize_flag,
    CASE WHEN sum(coalesce(tr.honsyokin, 0)::bigint) >= 100000 THEN 1 ELSE 0 END AS pog_total_prize_ge_10m_flag,
    CASE WHEN sum(coalesce(tr.honsyokin, 0)::bigint) >= 300000 THEN 1 ELSE 0 END AS pog_total_prize_ge_30m_flag

from horse_window hw
left join target_races tr
  on hw.ketto_num = tr.ketto_num
 and tr.race_date between hw.pog_start_date and hw.pog_end_date
group by
    hw.ketto_num,
    hw.birth_year,
    hw.is_jra_registered,
    hw.reg_date,
    hw.pog_start_date,
    hw.pog_end_date,
    hw.label_complete, hw.rule_comment;

create unique index if not exists idx_mv_horse_labels_ketto_num
    on pog.mv_horse_labels (ketto_num, rule_comment);

create index if not exists idx_mv_horse_labels_birth_year
    on pog.mv_horse_labels (birth_year, rule_comment, label_complete);

drop materialized view if exists pog.mv_sire_hist_stats;

create materialized view pog.mv_sire_hist_stats as
with hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.sire_hansyoku_num,
        hl.win_flag,
        hl.bt_place_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
    AND cc.rule_comment = 'arima'
)
select
    target.birth_year as target_birth_year,
    hist.sire_hansyoku_num,
    count(*) as sire_prior_foals,
    avg(hist.win_flag::numeric) as sire_prior_win_rate,
    avg(hist.bt_place_flag::numeric) as sire_prior_bt_rate,
    avg(hist.graded_win_flag::numeric) as sire_prior_graded_win_rate,
    avg(ln(1 + hist.pog_total_prize::numeric)) as sire_prior_avg_log_prize,
    percentile_cont(0.5) within group (order by hist.pog_total_prize) as sire_prior_med_prize
from pog.cohort_calendar target
join hist_horse hist
  on hist.birth_year < target.birth_year
group by
    target.birth_year,
    hist.sire_hansyoku_num;

create index if not exists idx_mv_sire_hist_stats_key
    on pog.mv_sire_hist_stats (target_birth_year, sire_hansyoku_num);

drop materialized view if exists pog.mv_dam_hist_stats;

create materialized view pog.mv_dam_hist_stats as
with hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.dam_hansyoku_num,
        hl.win_flag,
        hl.bt_place_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
      AND cc.rule_comment = 'arima'
)
select
    target.birth_year as target_birth_year,
    hist.dam_hansyoku_num,
    count(*) as dam_prior_foals,
    avg(hist.win_flag::numeric) as dam_prior_win_rate,
    avg(hist.bt_place_flag::numeric) as dam_prior_bt_rate,
    avg(hist.graded_win_flag::numeric) as dam_prior_graded_win_rate,
    avg(ln(1 + hist.pog_total_prize::numeric)) as dam_prior_avg_log_prize,
    percentile_cont(0.5) within group (order by hist.pog_total_prize) as dam_prior_med_prize
from pog.cohort_calendar target
join hist_horse hist
  on hist.birth_year < target.birth_year
group by
    target.birth_year,
    hist.dam_hansyoku_num;

create index if not exists idx_mv_dam_hist_stats_key
    on pog.mv_dam_hist_stats (target_birth_year, dam_hansyoku_num);

drop materialized view if exists pog.mv_breeder_hist_stats;

create materialized view pog.mv_breeder_hist_stats as
with hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.breeder_code,
        hl.win_flag,
        hl.bt_place_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
)
select
    target.birth_year as target_birth_year,
    hist.breeder_code,
    count(*) as breeder_prior_foals,
    avg(hist.win_flag::numeric) as breeder_prior_win_rate,
    avg(hist.bt_place_flag::numeric) as breeder_prior_bt_rate,
    avg(hist.graded_win_flag::numeric) as breeder_prior_graded_win_rate,
    avg(ln(1 + hist.pog_total_prize::numeric)) as breeder_prior_avg_log_prize,
    percentile_cont(0.5) within group (order by hist.pog_total_prize) as breeder_prior_med_prize
from pog.cohort_calendar target
join hist_horse hist
  on hist.birth_year < target.birth_year
group by
    target.birth_year,
    hist.breeder_code;

create index if not exists idx_mv_breeder_hist_stats_key
    on pog.mv_breeder_hist_stats (target_birth_year, breeder_code);

drop materialized view if exists pog.mv_trainer_hist_stats;

create materialized view pog.mv_trainer_hist_stats as
with hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.chokyosi_code,
        hl.win_flag,
        hl.bt_place_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
      and hm.chokyosi_code is not null
)
select
    target.birth_year as target_birth_year,
    hist.chokyosi_code,
    count(*) as trainer_prior_foals,
    avg(hist.win_flag::numeric) as trainer_prior_win_rate,
    avg(hist.bt_place_flag::numeric) as trainer_prior_bt_rate,
    avg(hist.graded_win_flag::numeric) as trainer_prior_graded_win_rate,
    avg(ln(1 + hist.pog_total_prize::numeric)) as trainer_prior_avg_log_prize,
    percentile_cont(0.5) within group (order by hist.pog_total_prize) as trainer_prior_med_prize
from pog.cohort_calendar target
join hist_horse hist
  on hist.birth_year < target.birth_year
group by
    target.birth_year,
    hist.chokyosi_code;

create index if not exists idx_mv_trainer_hist_stats_key
    on pog.mv_trainer_hist_stats (target_birth_year, chokyosi_code);

drop materialized view if exists pog.mv_static_features;

create materialized view pog.mv_static_features as
select
    hm.ketto_num,
    hm.birth_year,
    hm.birth_date,
    hm.birth_month,
    hm.sex_cd,
    hm.hinsyu_cd,
    hm.sanku_mochi_kubun,
    hm.import_year,
    hm.breeder_code,
    hm.sanchi_name,
    hm.is_jra_registered,
    hm.reg_date,
    hm.days_birth_to_reg,
    hm.tozai_cd,
    hm.chokyosi_code,
    hm.chokyosi_ryakusyo,
    hm.banusi_code,
    hm.banusi_name,
    hm.sire_hansyoku_num,
    hm.sire_name,
    hm.dam_hansyoku_num,
    hm.dam_name,
    hm.damsire_hansyoku_num,
    hm.damsire_name,

    ss.sire_prior_foals,
    ss.sire_prior_win_rate,
    ss.sire_prior_bt_rate,
    ss.sire_prior_graded_win_rate,
    ss.sire_prior_avg_log_prize,
    ss.sire_prior_med_prize,

    ds.dam_prior_foals,
    ds.dam_prior_win_rate,
    ds.dam_prior_bt_rate,
    ds.dam_prior_graded_win_rate,
    ds.dam_prior_avg_log_prize,
    ds.dam_prior_med_prize,

    bs.breeder_prior_foals,
    bs.breeder_prior_win_rate,
    bs.breeder_prior_bt_rate,
    bs.breeder_prior_graded_win_rate,
    bs.breeder_prior_avg_log_prize,
    bs.breeder_prior_med_prize,

    ts.trainer_prior_foals,
    ts.trainer_prior_win_rate,
    ts.trainer_prior_bt_rate,
    ts.trainer_prior_graded_win_rate,
    ts.trainer_prior_avg_log_prize,
    ts.trainer_prior_med_prize

from pog.mv_horse_master hm
left join pog.mv_sire_hist_stats ss
  on hm.birth_year = ss.target_birth_year
 and hm.sire_hansyoku_num = ss.sire_hansyoku_num
left join pog.mv_dam_hist_stats ds
  on hm.birth_year = ds.target_birth_year
 and hm.dam_hansyoku_num = ds.dam_hansyoku_num
left join pog.mv_breeder_hist_stats bs
  on hm.birth_year = bs.target_birth_year
 and hm.breeder_code = bs.breeder_code
left join pog.mv_trainer_hist_stats ts
  on hm.birth_year = ts.target_birth_year
 and hm.chokyosi_code = ts.chokyosi_code;

create unique index if not exists idx_mv_static_features_ketto_num
    on pog.mv_static_features (ketto_num);

create index if not exists idx_mv_static_features_birth_year
    on pog.mv_static_features (birth_year);

create or replace function pog.fn_dynamic_features(
    p_birth_year integer,
    p_asof_date date
)
returns table
(
    ketto_num              text,
    starts_to_asof         integer,
    wins_to_asof           integer,
    best_finish_to_asof    integer,
    avg_odds_to_asof       numeric,
    min_odds_to_asof       numeric,
    avg_ninki_to_asof      numeric,
    total_prize_to_asof    bigint,
    total_honsyokin_to_asof bigint,
    last_race_date         date,
    last_finish            integer,
    last_odds              numeric,
    last_ninki             integer,
    days_since_last_start  integer
)
language sql
as $$
with base as (
    select
        hm.ketto_num
    from pog.mv_horse_master hm
    where hm.birth_year = p_birth_year
),
r as (
    select
        re.*,
        row_number() over (
            partition by re.ketto_num
            order by re.race_date desc, re.jyo_cd desc, re.race_num desc
        ) as rn_desc
    from pog.mv_race_result_enriched re
    join base b
      on re.ketto_num = b.ketto_num
    where re.is_pog_target = true
      and re.race_date <= p_asof_date
)
select
    b.ketto_num,
    count(r.ketto_num)::int as starts_to_asof,
    sum(case when r.kakutei_jyuni = 1 then 1 else 0 end)::int as wins_to_asof,
    min(r.kakutei_jyuni)::int as best_finish_to_asof,
    avg(r.odds) as avg_odds_to_asof,
    min(r.odds) as min_odds_to_asof,
    avg(r.ninki) as avg_ninki_to_asof,
    coalesce(sum(coalesce(r.honsyokin, 0) + coalesce(r.fukasyokin, 0)), 0)::bigint as total_prize_to_asof,
    coalesce(sum(coalesce(r.honsyokin, 0)), 0)::bigint as total_honsyokin_to_asof,
    max(r.race_date) as last_race_date,
    max(case when r.rn_desc = 1 then r.kakutei_jyuni end)::int as last_finish,
    max(case when r.rn_desc = 1 then r.odds end) as last_odds,
    max(case when r.rn_desc = 1 then r.ninki end)::int as last_ninki,
    case
        when max(r.race_date) is null then null
        else (p_asof_date - max(r.race_date))::int
    end as days_since_last_start
from base b
left join r
  on b.ketto_num = r.ketto_num
group by b.ketto_num;
$$;

create table if not exists pog.model_predictions
(
    model_name            text        not null,
    model_version         text        not null,
    asof_date             date        not null,
    birth_year            integer     not null,
    ketto_num             text        not null,
    p_win                 double precision,
    p_bt_place            double precision,
    p_graded_win          double precision,
    p_positive_prize      double precision,
    expected_pog_prize    double precision,
    pred_log_prize_pos    double precision,
    created_at            timestamptz not null default now(),
    primary key (model_name, model_version, asof_date, ketto_num)
);

-- v2
drop materialized view if exists pog.mv_horse_master_ext;

create materialized view pog.mv_horse_master_ext as
select
    hm.*,
    hs.birth_year as sire_birth_year,
    hd.birth_year as dam_birth_year,
    hds.birth_year as damsire_birth_year,
    hd.hansyoku_dam_num as granddam_hansyoku_num,
    case
        when hd.birth_year is not null then hm.birth_year - hd.birth_year
        else null
    end as dam_age_at_foaling,
    case
        when hs.birth_year is not null then hm.birth_year - hs.birth_year
        else null
    end as sire_age_at_foaling
from pog.mv_horse_master hm
left join public.hansyokus hs
  on hm.sire_hansyoku_num = hs.hansyoku_num
left join public.hansyokus hd
  on hm.dam_hansyoku_num = hd.hansyoku_num
left join public.hansyokus hds
  on hm.damsire_hansyoku_num = hds.hansyoku_num;

create unique index if not exists idx_mv_horse_master_ext_ketto_num
    on pog.mv_horse_master_ext (ketto_num);

create index if not exists idx_mv_horse_master_ext_birth_year
    on pog.mv_horse_master_ext (birth_year);

create index if not exists idx_mv_horse_master_ext_granddam
    on pog.mv_horse_master_ext (granddam_hansyoku_num);

drop materialized view if exists pog.mv_damsire_hist_stats;

create materialized view pog.mv_damsire_hist_stats as
with hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.damsire_hansyoku_num,
        hl.win_flag,
        hl.bt_place_flag,
        hl.bt_win_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master_ext hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
      and hm.damsire_hansyoku_num is not null
)
select
    target.birth_year as target_birth_year,
    hist.damsire_hansyoku_num,
    count(*) as damsire_prior_foals,
    avg(hist.win_flag::numeric) as damsire_prior_win_rate,
    avg(hist.bt_place_flag::numeric) as damsire_prior_bt_place_rate,
    avg(hist.bt_win_flag::numeric) as damsire_prior_bt_win_rate,
    avg(hist.graded_win_flag::numeric) as damsire_prior_graded_win_rate,
    avg(ln(1 + hist.pog_total_prize::numeric)) as damsire_prior_avg_log_prize,
    percentile_cont(0.5) within group (order by hist.pog_total_prize) as damsire_prior_med_prize,
    max(hist.pog_total_prize) as damsire_prior_best_prize
from pog.cohort_calendar target
join hist_horse hist
  on hist.birth_year < target.birth_year
group by
    target.birth_year,
    hist.damsire_hansyoku_num;

create index if not exists idx_mv_damsire_hist_stats_key
    on pog.mv_damsire_hist_stats (target_birth_year, damsire_hansyoku_num);

drop materialized view if exists pog.mv_prior_maternal_sibling_stats;

create materialized view pog.mv_prior_maternal_sibling_stats as
with target_horse as (
    select
        ketto_num,
        birth_year,
        birth_date,
        dam_hansyoku_num
    from pog.mv_horse_master_ext
    where dam_hansyoku_num is not null
),
hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.birth_date,
        hm.dam_hansyoku_num,
        hl.win_flag,
        hl.bt_place_flag,
        hl.bt_win_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master_ext hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
)
select
    t.ketto_num,
    count(h.ketto_num) as prior_maternal_sib_count,
    sum(h.win_flag) as prior_maternal_sib_win_count,
    sum(h.bt_place_flag) as prior_maternal_sib_bt_place_count,
    sum(h.bt_win_flag) as prior_maternal_sib_bt_win_count,
    sum(h.graded_win_flag) as prior_maternal_sib_graded_win_count,
    avg(h.win_flag::numeric) as prior_maternal_sib_win_rate,
    avg(h.bt_place_flag::numeric) as prior_maternal_sib_bt_place_rate,
    avg(h.bt_win_flag::numeric) as prior_maternal_sib_bt_win_rate,
    avg(h.graded_win_flag::numeric) as prior_maternal_sib_graded_win_rate,
    avg(ln(1 + h.pog_total_prize::numeric)) as prior_maternal_sib_avg_log_prize,
    percentile_cont(0.5) within group (order by h.pog_total_prize) as prior_maternal_sib_med_prize,
    max(h.pog_total_prize) as prior_maternal_sib_best_prize
from target_horse t
left join hist_horse h
  on t.dam_hansyoku_num = h.dam_hansyoku_num
 and h.birth_year < t.birth_year
group by t.ketto_num;

create unique index if not exists idx_mv_prior_maternal_sibling_stats_ketto_num
    on pog.mv_prior_maternal_sibling_stats (ketto_num);

drop materialized view if exists pog.mv_prior_full_sibling_stats;

create materialized view pog.mv_prior_full_sibling_stats as
with target_horse as (
    select
        ketto_num,
        birth_year,
        sire_hansyoku_num,
        dam_hansyoku_num
    from pog.mv_horse_master_ext
    where sire_hansyoku_num is not null
      and dam_hansyoku_num is not null
),
hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.sire_hansyoku_num,
        hm.dam_hansyoku_num,
        hl.win_flag,
        hl.bt_place_flag,
        hl.bt_win_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master_ext hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
)
select
    t.ketto_num,
    count(h.ketto_num) as prior_full_sib_count,
    avg(h.win_flag::numeric) as prior_full_sib_win_rate,
    avg(h.bt_place_flag::numeric) as prior_full_sib_bt_place_rate,
    avg(h.bt_win_flag::numeric) as prior_full_sib_bt_win_rate,
    avg(h.graded_win_flag::numeric) as prior_full_sib_graded_win_rate,
    avg(ln(1 + h.pog_total_prize::numeric)) as prior_full_sib_avg_log_prize,
    percentile_cont(0.5) within group (order by h.pog_total_prize) as prior_full_sib_med_prize,
    max(h.pog_total_prize) as prior_full_sib_best_prize
from target_horse t
left join hist_horse h
  on t.sire_hansyoku_num = h.sire_hansyoku_num
 and t.dam_hansyoku_num = h.dam_hansyoku_num
 and h.birth_year < t.birth_year
group by t.ketto_num;

create unique index if not exists idx_mv_prior_full_sibling_stats_ketto_num
    on pog.mv_prior_full_sibling_stats (ketto_num);

drop materialized view if exists pog.mv_sire_damsire_nick_stats;

create materialized view pog.mv_sire_damsire_nick_stats as
with hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.sire_hansyoku_num,
        hm.damsire_hansyoku_num,
        hl.win_flag,
        hl.bt_place_flag,
        hl.bt_win_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master_ext hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
      and hm.sire_hansyoku_num is not null
      and hm.damsire_hansyoku_num is not null
)
select
    target.birth_year as target_birth_year,
    hist.sire_hansyoku_num,
    hist.damsire_hansyoku_num,
    count(*) as nick_prior_foals,
    avg(hist.win_flag::numeric) as nick_prior_win_rate,
    avg(hist.bt_place_flag::numeric) as nick_prior_bt_place_rate,
    avg(hist.bt_win_flag::numeric) as nick_prior_bt_win_rate,
    avg(hist.graded_win_flag::numeric) as nick_prior_graded_win_rate,
    avg(ln(1 + hist.pog_total_prize::numeric)) as nick_prior_avg_log_prize,
    max(hist.pog_total_prize) as nick_prior_best_prize
from pog.cohort_calendar target
join hist_horse hist
  on hist.birth_year < target.birth_year
group by
    target.birth_year,
    hist.sire_hansyoku_num,
    hist.damsire_hansyoku_num;

create index if not exists idx_mv_sire_damsire_nick_stats_key
    on pog.mv_sire_damsire_nick_stats (
        target_birth_year,
        sire_hansyoku_num,
        damsire_hansyoku_num
    );

drop materialized view if exists pog.mv_breeder_trainer_hist_stats;

create materialized view pog.mv_breeder_trainer_hist_stats as
with hist_horse as (
    select
        hm.ketto_num,
        hm.birth_year,
        hm.breeder_code,
        hm.chokyosi_code,
        hl.win_flag,
        hl.bt_place_flag,
        hl.bt_win_flag,
        hl.graded_win_flag,
        hl.pog_total_prize
    from pog.mv_horse_master_ext hm
    join pog.mv_horse_labels hl
      on hm.ketto_num = hl.ketto_num
    join pog.cohort_calendar cc
      on hm.birth_year = cc.birth_year
    where cc.label_complete = true
      and hm.breeder_code is not null
      and hm.chokyosi_code is not null
)
select
    target.birth_year as target_birth_year,
    hist.breeder_code,
    hist.chokyosi_code,
    count(*) as breeder_trainer_prior_foals,
    avg(hist.win_flag::numeric) as breeder_trainer_prior_win_rate,
    avg(hist.bt_place_flag::numeric) as breeder_trainer_prior_bt_place_rate,
    avg(hist.bt_win_flag::numeric) as breeder_trainer_prior_bt_win_rate,
    avg(hist.graded_win_flag::numeric) as breeder_trainer_prior_graded_win_rate,
    avg(ln(1 + hist.pog_total_prize::numeric)) as breeder_trainer_prior_avg_log_prize,
    max(hist.pog_total_prize) as breeder_trainer_prior_best_prize
from pog.cohort_calendar target
join hist_horse hist
  on hist.birth_year < target.birth_year
group by
    target.birth_year,
    hist.breeder_code,
    hist.chokyosi_code;

create index if not exists idx_mv_breeder_trainer_hist_stats_key
    on pog.mv_breeder_trainer_hist_stats (
        target_birth_year,
        breeder_code,
        chokyosi_code
    );

drop materialized view if exists pog.mv_static_features_v2;

create materialized view pog.mv_static_features_v2 as
select
    hm.ketto_num,
    hm.birth_year,
    hm.birth_date,
    hm.birth_month,
    hm.sex_cd,
    hm.hinsyu_cd,
    hm.sanku_mochi_kubun,
    hm.import_year,
    hm.breeder_code,
    hm.sanchi_name,
    hm.is_jra_registered,
    hm.reg_date,
    hm.days_birth_to_reg,
    hm.tozai_cd,
    hm.chokyosi_code,
    hm.chokyosi_ryakusyo,
    hm.banusi_code,
    hm.banusi_name,
    hm.sire_hansyoku_num,
    hm.sire_name,
    hm.dam_hansyoku_num,
    hm.dam_name,
    hm.damsire_hansyoku_num,
    hm.damsire_name,
    hm.granddam_hansyoku_num,
    hm.dam_age_at_foaling,
    hm.sire_age_at_foaling,

    -- existing sire / dam / breeder / trainer stats
    ss.sire_prior_foals,
    ss.sire_prior_win_rate,
    ss.sire_prior_bt_rate,
    ss.sire_prior_graded_win_rate,
    ss.sire_prior_avg_log_prize,
    ss.sire_prior_med_prize,

    ds.dam_prior_foals,
    ds.dam_prior_win_rate,
    ds.dam_prior_bt_rate,
    ds.dam_prior_graded_win_rate,
    ds.dam_prior_avg_log_prize,
    ds.dam_prior_med_prize,

    bs.breeder_prior_foals,
    bs.breeder_prior_win_rate,
    bs.breeder_prior_bt_rate,
    bs.breeder_prior_graded_win_rate,
    bs.breeder_prior_avg_log_prize,
    bs.breeder_prior_med_prize,

    ts.trainer_prior_foals,
    ts.trainer_prior_win_rate,
    ts.trainer_prior_bt_rate,
    ts.trainer_prior_graded_win_rate,
    ts.trainer_prior_avg_log_prize,
    ts.trainer_prior_med_prize,

    -- new damsire stats
    dss.damsire_prior_foals,
    dss.damsire_prior_win_rate,
    dss.damsire_prior_bt_place_rate,
    dss.damsire_prior_bt_win_rate,
    dss.damsire_prior_graded_win_rate,
    dss.damsire_prior_avg_log_prize,
    dss.damsire_prior_med_prize,
    dss.damsire_prior_best_prize,

    -- new sibling stats
    ms.prior_maternal_sib_count,
    ms.prior_maternal_sib_win_count,
    ms.prior_maternal_sib_bt_place_count,
    ms.prior_maternal_sib_bt_win_count,
    ms.prior_maternal_sib_graded_win_count,
    ms.prior_maternal_sib_win_rate,
    ms.prior_maternal_sib_bt_place_rate,
    ms.prior_maternal_sib_bt_win_rate,
    ms.prior_maternal_sib_graded_win_rate,
    ms.prior_maternal_sib_avg_log_prize,
    ms.prior_maternal_sib_med_prize,
    ms.prior_maternal_sib_best_prize,

    fs.prior_full_sib_count,
    fs.prior_full_sib_win_rate,
    fs.prior_full_sib_bt_place_rate,
    fs.prior_full_sib_bt_win_rate,
    fs.prior_full_sib_graded_win_rate,
    fs.prior_full_sib_avg_log_prize,
    fs.prior_full_sib_med_prize,
    fs.prior_full_sib_best_prize,

    -- nick stats
    ns.nick_prior_foals,
    ns.nick_prior_win_rate,
    ns.nick_prior_bt_place_rate,
    ns.nick_prior_bt_win_rate,
    ns.nick_prior_graded_win_rate,
    ns.nick_prior_avg_log_prize,
    ns.nick_prior_best_prize,

    -- combo stats
    bts.breeder_trainer_prior_foals,
    bts.breeder_trainer_prior_win_rate,
    bts.breeder_trainer_prior_bt_place_rate,
    bts.breeder_trainer_prior_bt_win_rate,
    bts.breeder_trainer_prior_graded_win_rate,
    bts.breeder_trainer_prior_avg_log_prize,
    bts.breeder_trainer_prior_best_prize

from pog.mv_horse_master_ext hm
left join pog.mv_sire_hist_stats ss
  on hm.birth_year = ss.target_birth_year
 and hm.sire_hansyoku_num = ss.sire_hansyoku_num
left join pog.mv_dam_hist_stats ds
  on hm.birth_year = ds.target_birth_year
 and hm.dam_hansyoku_num = ds.dam_hansyoku_num
left join pog.mv_breeder_hist_stats bs
  on hm.birth_year = bs.target_birth_year
 and hm.breeder_code = bs.breeder_code
left join pog.mv_trainer_hist_stats ts
  on hm.birth_year = ts.target_birth_year
 and hm.chokyosi_code = ts.chokyosi_code
left join pog.mv_damsire_hist_stats dss
  on hm.birth_year = dss.target_birth_year
 and hm.damsire_hansyoku_num = dss.damsire_hansyoku_num
left join pog.mv_prior_maternal_sibling_stats ms
  on hm.ketto_num = ms.ketto_num
left join pog.mv_prior_full_sibling_stats fs
  on hm.ketto_num = fs.ketto_num
left join pog.mv_sire_damsire_nick_stats ns
  on hm.birth_year = ns.target_birth_year
 and hm.sire_hansyoku_num = ns.sire_hansyoku_num
 and hm.damsire_hansyoku_num = ns.damsire_hansyoku_num
left join pog.mv_breeder_trainer_hist_stats bts
  on hm.birth_year = bts.target_birth_year
 and hm.breeder_code = bts.breeder_code
 and hm.chokyosi_code = bts.chokyosi_code;

create unique index if not exists idx_mv_static_features_v2_ketto_num
    on pog.mv_static_features_v2 (ketto_num);

create index if not exists idx_mv_static_features_v2_birth_year
    on pog.mv_static_features_v2 (birth_year);
