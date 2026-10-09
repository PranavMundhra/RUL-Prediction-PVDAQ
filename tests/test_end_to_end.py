# SPDX-License-Identifier: Apache-2.0
"""Whole library chain on a fake S3: inventory -> profile -> verdicts."""

import pandas as pd

from pvdaq_rul import io
from pvdaq_rul.quality import profile_days
from pvdaq_rul.usability import assign_reasons, build_daily, flag_snow, year_scorecard
from tests.fake_pvdaq import make_day, metric_names


class FakeFS:
    def __init__(self, files):
        self.files = files

    def find(self, root, detail=False):
        return {p: {"type": "file", "size": len(df)} for p, df in self.files.items() if p.startswith(root)}


def test_chain_inventory_to_scorecard(cfg, monkeypatch):
    files = {}
    for d in pd.date_range("2021-01-01", "2021-12-31", freq="2D"):
        p = f"{io.BASE}/pvdata/system_id=50/year={d.year}/month={d.month}/day={d.day}/part-0.parquet"
        files[p] = make_day(d, seed=int(d.dayofyear))
    bad = "2021-05-05"
    files[next(p for p in files if "month=5/day=5/" in p)] = None  # unreadable day -> recorded, not fatal

    def fake_read(path, filesystem=None, columns=None, **kw):
        df = files[path]
        if df is None:
            raise OSError("corrupt file")
        return df[columns].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    fake_files = {p: (df if df is not None else make_day(bad)) for p, df in files.items()}
    fs = FakeFS(fake_files)

    days = io.group_days(io.build_inventory(fs, 50))
    prof = profile_days(days, fs, metric_names(), cfg, workers=4)
    assert prof["read_error"].notna().sum() == 1

    d = assign_reasons(flag_snow(build_daily(prof), cfg), cfg)
    assert d.loc[d["date"] == bad, "reason"].iloc[0] == "no_data"
    sc = year_scorecard(d, cfg)
    assert sc.loc[0, "verdict"] == "USE"
