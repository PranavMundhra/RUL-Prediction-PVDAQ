# SPDX-License-Identifier: Apache-2.0
import pandas as pd

from pvdaq_rul import Config
from pvdaq_rul.usability import REASONS, assign_reasons, build_daily, flag_snow, waterfall, year_scorecard
from tests.fake_pvdaq import make_profiled


def test_snow_episode_is_detected_and_dropped(cfg, names):
    snow = pd.date_range("2021-02-09", "2021-02-20", freq="D")
    prof = make_profiled(cfg, names, "2021-01-01", "2021-03-31", step="1D", snow_days=set(snow))
    d = assign_reasons(flag_snow(build_daily(prof), cfg), cfg)
    hit = d[d["date"].isin(snow)]
    assert hit["snow_episode"].mean() > 0.8
    assert (hit["reason"] == "snow_episode").mean() > 0.8
    assert d[d["date"] < "2021-02-01"]["usable"].all()


def test_manual_exclusion_wins_over_data(names):
    cfg = Config(exclude_periods=(("2021-01-10", "2021-01-12", "test storm"),))
    prof = make_profiled(cfg, names, "2021-01-01", "2021-01-20", step="1D")
    d = assign_reasons(flag_snow(build_daily(prof), cfg), cfg)
    ex = d[d["date"].between("2021-01-10", "2021-01-12")]
    assert (ex["reason"] == "manual_exclusion").all()
    assert (ex["manual_exclusion_why"] == "test storm").all()


def test_conflicts_are_blamed_on_conflicts_not_coverage(cfg, names):
    days = set(pd.date_range("2021-03-01", "2021-03-05", freq="D"))
    prof = make_profiled(cfg, names, "2021-03-01", "2021-03-10", step="1D", conflict_days=days)
    d = assign_reasons(flag_snow(build_daily(prof), cfg), cfg)
    assert (d.loc[d["date"].isin(days), "reason"] == "conflicts").all()


def test_missing_days_appear_as_no_data(cfg, names):
    prof = make_profiled(cfg, names, "2021-04-01", "2021-04-20", step="5D")
    d = assign_reasons(flag_snow(build_daily(prof), cfg), cfg)
    assert len(d) == 16 and (d["reason"] == "no_data").sum() == 12


def test_waterfall_adds_up(cfg, names):
    prof = make_profiled(cfg, names, "2021-04-01", "2021-04-20", step="2D")
    d = assign_reasons(flag_snow(build_daily(prof), cfg), cfg)
    wf = waterfall(d)
    assert wf["days_left"].iloc[0] == len(d)
    assert wf["days_left"].iloc[-1] == int(d["usable"].sum())
    assert set(REASONS[1:]) <= set(wf["step"].str.replace("after rule: ", ""))


def test_year_verdicts(cfg, names):
    prof = make_profiled(cfg, names, "2021-01-01", "2021-12-31", step="2D")
    d = assign_reasons(flag_snow(build_daily(prof), cfg), cfg)
    sc = year_scorecard(d, cfg)
    assert sc.loc[0, "verdict"] == "USE" and sc.loc[0, "usable_days"] >= cfg.use_min_days
    sparse = make_profiled(cfg, names, "2022-06-01", "2022-06-30", step="3D")
    sc2 = year_scorecard(assign_reasons(flag_snow(build_daily(sparse), cfg), cfg), cfg)
    assert sc2.loc[0, "verdict"] == "DO_NOT_USE"
