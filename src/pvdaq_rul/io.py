# SPDX-License-Identifier: Apache-2.0
"""Reading PVDAQ data from the public OEDI S3 bucket."""

from __future__ import annotations

import pandas as pd

BUCKET = "oedi-data-lake"
BASE = f"{BUCKET}/pvdaq/parquet"
KEYS = ["measured_on", "metric_id"]
RAW_COLUMNS = ["measured_on", "utc_measured_on", "metric_id", "value"]


def make_fs():
    """Anonymous S3 filesystem (no credentials needed for the public bucket)."""
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def load_metric_names(fs, system_id: int) -> dict[int, str]:
    """Map metric_id -> sensor_name for one system."""
    metrics = pd.read_parquet(f"{BASE}/metrics", filesystem=fs)
    m = metrics[metrics["system_id"].eq(system_id)].sort_values("metric_id")
    if not m["metric_id"].is_unique:
        raise ValueError("metric_id is not unique in the metadata; inspect it first")
    return m.set_index("metric_id")["sensor_name"].to_dict()


def build_inventory(fs, system_id: int) -> pd.DataFrame:
    """List every data file of a system (no downloading) with year/month/day parsed."""
    root = f"{BASE}/pvdata/system_id={system_id}"
    info = fs.find(root, detail=True)
    rows = [
        (p, v.get("size", 0))
        for p, v in info.items()
        if v.get("type") != "directory" and v.get("size", 0) > 0
    ]
    inv = pd.DataFrame(rows, columns=["path", "size"])
    inv = inv[~inv["path"].str.rsplit("/", n=1).str[-1].str.startswith(("_", "."))]
    ex = inv["path"].str.extract(r"year=(\d+)/month=(\d+)/day=(\d+)")
    ok = ex.notna().all(axis=1)
    inv = inv[ok].copy()
    inv[["year", "month", "day"]] = ex[ok].astype(int).to_numpy()
    inv["date"] = pd.to_datetime(inv[["year", "month", "day"]])
    inv["day_dir"] = inv["path"].str.rsplit("/", n=1).str[0]
    return inv.sort_values("path", ignore_index=True)


def group_days(inventory: pd.DataFrame) -> pd.DataFrame:
    """One row per day: date, number of files, size, list of file paths."""
    return (
        inventory.groupby("day_dir")
        .agg(
            date=("date", "first"),
            n_files=("path", "size"),
            mb=("size", lambda s: s.sum() / 1e6),
            paths=("path", list),
        )
        .reset_index(drop=True)
        .sort_values("date", ignore_index=True)
    )


def read_day(fs, paths: list[str]) -> pd.DataFrame:
    """Read all files of one day; adds file_no so duplicate ingestion can be traced."""
    frames = []
    for k, p in enumerate(paths):
        d = pd.read_parquet(p, filesystem=fs, columns=RAW_COLUMNS)
        d["file_no"] = k
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
