# SPDX-License-Identifier: Apache-2.0
"""Per-day quality profiling: duplicates, conflicts, coverage, POA validity, health index."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pvlib

from .config import Config
from .health_index import clearsky_poa, performance_index
from .io import KEYS, read_day

MINUTES = 1440


def longest_run(a) -> int:
    """Length of the longest streak of True values."""
    a = np.asarray(a, dtype=np.int8)
    if not a.any():
        return 0
    d = np.diff(np.concatenate(([0], a, [0])))
    return int((np.where(d == -1)[0] - np.where(d == 1)[0]).max())


def clean_day(raw: pd.DataFrame, date: pd.Timestamp, names: dict, cfg: Config):
    """Return (wide 1-minute table, per-minute info, utc index, counts).

    Exact duplicate copies are collapsed to one. Conflicting copies (same minute and
    sensor, different values) are set to missing, never averaged.
    """
    g = raw.groupby(KEYS)["value"].agg(n="size", nu="nunique")
    conflict_keys = g[g["nu"] > 1]
    conf_per_ts = (
        pd.Series(1, index=conflict_keys.index.get_level_values("measured_on")).groupby(level=0).size()
    )

    first = raw.drop_duplicates(KEYS, keep="first").merge(g["nu"].reset_index(), on=KEYS, how="left")
    first.loc[first["nu"] > 1, "value"] = np.nan

    wide = first.pivot(index="measured_on", columns="metric_id", values="value")
    wide.columns = [names.get(c, f"metric_{c}") for c in wide.columns]

    idx = pd.date_range(date, periods=MINUTES, freq="1min")
    n_outside = int((~wide.index.isin(idx)).sum())
    wide = wide.reindex(idx)

    off = (raw["utc_measured_on"] - raw["measured_on"]).mode().iloc[0]
    idx_utc = pd.DatetimeIndex(idx + off).tz_localize("UTC")
    sp = pvlib.solarposition.get_solarposition(idx_utc, cfg.lat, cfg.lon, altitude=cfg.alt)

    info = pd.DataFrame(
        {
            "elev": sp["apparent_elevation"].to_numpy(),
            "zen": sp["apparent_zenith"].to_numpy(),
            "az": sp["azimuth"].to_numpy(),
            "conflict_minute": idx.isin(conf_per_ts.index),
        },
        index=idx,
    )
    info["n_conflict_metrics"] = conf_per_ts.reindex(idx).fillna(0).astype(int).to_numpy()

    counts = {
        "n_rows": len(raw),
        "n_timestamps": int(raw["measured_on"].nunique()),
        "n_files": int(raw["file_no"].nunique()) if "file_no" in raw else 1,
        "n_ts_outside_day": n_outside,
        "utc_offset_h": off.total_seconds() / 3600,
        "exact_dup_groups": int(((g["n"] > 1) & (g["nu"] == 1)).sum()),
        "conflict_groups": int((g["nu"] > 1).sum()),
        "conflict_minutes_any": int(len(conf_per_ts)),
        "conflict_minutes_broad": int((conf_per_ts >= 7).sum()),
    }
    return wide, info, idx_utc, counts


def profile_day(date: pd.Timestamp, raw: pd.DataFrame, names: dict, cfg: Config) -> dict:
    """One row of quality measurements for one day of raw data."""
    row = {"date": date, "year": date.year, "month": date.month, "doy": date.dayofyear}
    wide, info, idx_utc, counts = clean_day(raw, date, names, cfg)
    row.update(counts)

    def col(n):
        return wide[n] if n in wide else pd.Series(np.nan, index=wide.index)

    P, Q = col(cfg.power_channel), col("poa_irradiance")
    Tm, Ta = col("module_temp_1"), col("ambient_temp")
    Tm = Tm.where(Tm.between(-50, 100))
    Ta = Ta.where(Ta.between(-50, 60))
    day = (info["elev"] > cfg.sun_elev_day).to_numpy()
    night = (info["elev"] < -6).to_numpy()
    D = int(day.sum())
    row["daylight_minutes"] = D

    row["cov_day_power"] = float(P[day].notna().mean()) if D else np.nan
    row["max_daylight_gap_min"] = longest_run(P.isna().to_numpy() & day)
    row["conflict_share_day"] = float(info["conflict_minute"].to_numpy()[day].mean()) if D else np.nan
    row["power_flatline"] = bool(D and P[day].nunique() <= 3)
    row["dc_peak_w"] = float(P.quantile(0.999)) if P.notna().any() else np.nan
    row["dc_energy_kwh_obs"] = float(P.clip(lower=0).sum() / 60 / 1000)
    row["power_neg_day_frac"] = float((P[day] < -1).sum() / max(1, P[day].notna().sum())) if D else np.nan

    row["module_temp_day_mean"] = float(Tm[day].mean()) if D else np.nan
    row["ambient_day_mean"] = float(Ta[day].mean()) if D else np.nan
    row["ambient_min"] = float(Ta.min()) if Ta.notna().any() else np.nan

    row["poa_cov_day"] = float(Q[day].notna().mean()) if D else np.nan
    row["poa_cov_night"] = float(Q[night].notna().mean()) if night.any() else np.nan
    row["poa_day_median"] = float(Q[day].median()) if D else np.nan
    row["poa_night_median"] = float(Q[night].median()) if night.any() else np.nan
    both = Q[day].notna() & P[day].notna()
    row["poa_corr_power_day"] = float(Q[day][both].corr(P[day][both])) if both.sum() >= 30 else np.nan
    night_med = row["poa_night_median"] if pd.notna(row["poa_night_median"]) else 0
    corr = row["poa_corr_power_day"] if pd.notna(row["poa_corr_power_day"]) else 0
    row["poa_valid_day"] = bool(
        D
        and pd.notna(row["poa_day_median"])
        and row["poa_cov_day"] >= cfg.poa_min_day_cov
        and cfg.poa_day_median_range[0] <= row["poa_day_median"] <= cfg.poa_day_median_range[1]
        and cfg.poa_night_median_range[0] <= night_med <= cfg.poa_night_median_range[1]
        and corr >= cfg.poa_min_corr
    )

    poa_cs = clearsky_poa(info, idx_utc, cfg)
    row["cs_energy_kwh_m2"] = float(poa_cs.sum() / 60 / 1000)
    row.update(performance_index(P, Tm, info, poa_cs, cfg))
    return row


def safe_profile_day(date, paths, fs, names, cfg: Config) -> dict:
    """Like profile_day but reads the files and records errors instead of raising."""
    try:
        return profile_day(date, read_day(fs, paths), names, cfg)
    except Exception as e:  # noqa: BLE001 - one bad day must not stop a 30-year run
        return {
            "date": date,
            "year": date.year,
            "month": date.month,
            "doy": date.dayofyear,
            "read_error": f"{type(e).__name__}: {e}"[:200],
        }


def profile_days(days: pd.DataFrame, fs, names: dict, cfg: Config, workers: int = 16) -> pd.DataFrame:
    """Profile many days in parallel. `days` needs columns `date` and `paths`."""
    with ThreadPoolExecutor(workers) as ex:
        rows = list(
            ex.map(lambda t: safe_profile_day(t[0], t[1], fs, names, cfg), zip(days["date"], days["paths"]))
        )
    out = pd.DataFrame(rows).sort_values("date", ignore_index=True)
    if "read_error" not in out:
        out["read_error"] = None
    return out
