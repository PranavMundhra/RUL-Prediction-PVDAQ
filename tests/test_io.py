# SPDX-License-Identifier: Apache-2.0
import pandas as pd

from pvdaq_rul import io
from tests.fake_pvdaq import make_day


class FakeFS:
    def __init__(self, files):
        self.files = files

    def find(self, root, detail=False):
        return {p: {"type": "file", "size": len(df)} for p, df in self.files.items() if p.startswith(root)}


def _path(d, part="part-0.parquet"):
    return f"{io.BASE}/pvdata/system_id=50/year={d.year}/month={d.month}/day={d.day}/{part}"


def test_inventory_groups_days_and_ignores_bookkeeping_files(monkeypatch):
    d1, d2 = pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-02")
    files = {
        _path(d1): make_day(d1),
        _path(d1, "part-1.parquet"): make_day(d1),
        _path(d2): make_day(d2),
        _path(d2, "_SUCCESS"): make_day(d2),
    }
    inv = io.build_inventory(FakeFS(files), 50)
    days = io.group_days(inv)
    assert len(days) == 2 and days["n_files"].tolist() == [2, 1]
    assert not inv["path"].str.endswith("_SUCCESS").any()


def test_read_day_marks_file_numbers(monkeypatch):
    d = pd.Timestamp("2021-06-01")
    files = {_path(d): make_day(d, seed=1), _path(d, "part-1.parquet"): make_day(d, seed=1)}
    monkeypatch.setattr(
        pd, "read_parquet", lambda p, filesystem=None, columns=None, **kw: files[p][columns].copy()
    )
    raw = io.read_day(None, list(files))
    assert raw["file_no"].nunique() == 2
