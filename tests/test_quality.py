# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd

from pvdaq_rul.quality import clean_day, longest_run, profile_day
from tests.fake_pvdaq import make_day


def test_longest_run():
    assert longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    assert longest_run([0, 0]) == 0


def test_exact_duplicates_are_collapsed_not_averaged(cfg, names):
    date = pd.Timestamp("2021-06-15")
    raw = make_day(date, seed=1)
    doubled = pd.concat([raw, raw], ignore_index=True)
    wide, _, _, counts = clean_day(doubled, date, names, cfg)
    wide_single, _, _, _ = clean_day(raw, date, names, cfg)
    assert counts["exact_dup_groups"] > 0 and counts["conflict_groups"] == 0
    pd.testing.assert_frame_equal(wide, wide_single)


def test_conflicting_copies_become_missing(cfg, names):
    date = pd.Timestamp("2021-06-15")
    raw = make_day(date, conflict=True, seed=1)
    wide, info, _, counts = clean_day(raw, date, names, cfg)
    assert counts["conflict_groups"] > 0
    conflict_minutes = info.index[info["conflict_minute"]]
    assert wide.loc[conflict_minutes, "dc_power"].isna().all()  # never averaged
    assert info["conflict_minute"].sum() == counts["conflict_minutes_any"]


def test_clean_day_has_full_minute_index(cfg, names):
    wide, info, idx_utc, counts = clean_day(
        make_day("2021-06-15", seed=2), pd.Timestamp("2021-06-15"), names, cfg
    )
    assert len(wide) == 1440 and len(info) == 1440 and idx_utc.tz is not None
    assert counts["utc_offset_h"] == 7.0


def test_profile_clean_summer_day(cfg, names):
    date = pd.Timestamp("2021-06-15")
    row = profile_day(date, make_day(date, seed=3), names, cfg)
    assert row["cov_day_power"] > 0.99
    assert row["max_daylight_gap_min"] == 0
    assert row["conflict_share_day"] == 0
    assert row["clear_minutes"] >= cfg.min_clear_minutes
    assert np.isfinite(row["perf_index_tc"])
    assert row["poa_valid_day"] is False  # synthetic default: POA only logged at night


def test_poa_validity_detects_a_working_sensor(cfg, names):
    date = pd.Timestamp("2021-06-15")
    good = profile_day(date, make_day(date, poa_ok=True, seed=4), names, cfg)
    bad = profile_day(date, make_day(date, poa_ok=False, seed=4), names, cfg)
    assert good["poa_valid_day"] is True
    assert bad["poa_valid_day"] is False


def test_conflict_day_has_large_conflict_share(cfg, names):
    date = pd.Timestamp("2021-06-15")
    row = profile_day(date, make_day(date, conflict=True, seed=5), names, cfg)
    assert row["conflict_share_day"] > cfg.max_conflict_share
