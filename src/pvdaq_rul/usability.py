# SPDX-License-Identifier: Apache-2.0
"""Turn per-day profiles into verdicts: which days and years are usable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .quality import longest_run

# Order matters: the first failing rule is the recorded reason.
REASONS = [
    "usable",
    "no_data",
    "manual_exclusion",
    "conflicts",
    "low_daylight_coverage",
    "long_gap",
    "flatline_or_bad_power",
    "snow_episode",
    "no_clear_minutes",
]


def build_daily(profiled: pd.DataFrame) -> pd.DataFrame:
    """Join profiles onto a full calendar so missing days appear explicitly."""
    cal = pd.DataFrame({"date": pd.date_range(profiled["date"].min(), profiled["date"].max(), freq="D")})
    daily = cal.merge(profiled.drop(columns=["year", "month", "doy"], errors="ignore"), on="date", how="left")
    if "read_error" not in daily:
        daily["read_error"] = None
    daily["has_file"] = daily["n_rows"].notna() & daily["read_error"].isna()
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["doy"] = daily["date"].dt.dayofyear
    return daily


def flag_snow(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add yield_idx, energy_vs_expected, snow_suspect and snow_episode columns."""
    d = daily.copy()
    good = d["cov_day_power"] >= cfg.min_daylight_cov
    d["yield_idx"] = (d["dc_energy_kwh_obs"] / d["cs_energy_kwh_m2"]).where(
        good & (d["cs_energy_kwh_m2"] > 0)
    )
    ref = d.set_index("date")["yield_idx"].rolling("365D", center=True, min_periods=30).quantile(0.90)
    d["yield_ref"] = ref.to_numpy()
    d["energy_vs_expected"] = d["yield_idx"] / d["yield_ref"]
    d["snow_suspect"] = (
        (d["energy_vs_expected"] < cfg.snow_energy_ratio) & (d["ambient_day_mean"] < cfg.snow_ambient_c)
    ).fillna(False)
    grp = (d["snow_suspect"] != d["snow_suspect"].shift()).cumsum()
    runlen = d["snow_suspect"].groupby(grp).transform("size")
    episode = d["snow_suspect"] & (runlen >= cfg.snow_min_run_days)
    d["snow_episode"] = episode.astype(int).rolling(cfg.snow_pad_days + 1, min_periods=1).max().astype(bool)
    return d


def assign_reasons(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Give every day one verdict (`reason`) and a boolean `usable`."""
    d = daily.copy()
    manual = pd.Series(False, index=d.index)
    why = pd.Series("", index=d.index)
    for start, end, text in cfg.exclude_periods:
        m = d["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        manual |= m
        why = why.mask(m, text)
    if cfg.exclude_years:
        m = d["year"].isin(cfg.exclude_years)
        manual |= m
        why = why.mask(m, "whole year excluded in settings")
    d["manual_exclusion"] = manual
    d["manual_exclusion_why"] = why

    conds = [
        ~d["has_file"],
        d["manual_exclusion"],
        ~(d["conflict_share_day"] <= cfg.max_conflict_share),
        ~(d["cov_day_power"] >= cfg.min_daylight_cov),
        ~(d["max_daylight_gap_min"] <= cfg.max_daylight_gap_min),
        d["power_flatline"].fillna(True).astype(bool) | ~(d["dc_peak_w"] > 0),
        d["snow_episode"],
        ~(d["clear_minutes"] >= cfg.min_clear_minutes),
    ]
    d["reason"] = np.select(conds, REASONS[1:], default="usable")
    d["usable"] = d["reason"].eq("usable")
    return d


def waterfall(d: pd.DataFrame) -> pd.DataFrame:
    """Days left after each rule, in rule order."""
    remaining = len(d)
    rows = [("all calendar days", remaining, 0)]
    for r in REASONS[1:]:
        n = int((d["reason"] == r).sum())
        remaining -= n
        rows.append((f"after rule: {r}", remaining, n))
    return pd.DataFrame(rows, columns=["step", "days_left", "dropped_by_rule"])


def year_scorecard(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per year with usable-day counts and a USE / USE_WITH_CAUTION / DO_NOT_USE verdict."""
    rows = []
    for y, g in d.groupby("year"):
        u = g[g["usable"]]
        months = int((u.groupby("month").size() >= 10).sum())
        n = len(u)
        if n >= cfg.use_min_days and months >= cfg.use_min_months:
            verdict = "USE"
        elif n >= cfg.caution_min_days:
            verdict = "USE_WITH_CAUTION"
        else:
            verdict = "DO_NOT_USE"
        drops = g.loc[~g["usable"], "reason"].value_counts().to_dict()
        rows.append(
            {
                "year": y,
                "calendar_days": len(g),
                "days_with_files": int(g["has_file"].sum()),
                "usable_days": n,
                "months_covered": months,
                "longest_usable_run_days": longest_run(g["usable"].to_numpy()),
                "verdict": verdict,
                **{f"drop_{r}": int(drops.get(r, 0)) for r in REASONS[1:]},
            }
        )
    return pd.DataFrame(rows)
