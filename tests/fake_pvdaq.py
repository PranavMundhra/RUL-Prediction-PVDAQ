# SPDX-License-Identifier: Apache-2.0
"""Synthetic System 50 data for tests (no internet needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

NAMES = [
    "poa_irradiance",
    "dc_power",
    "ac_power",
    "dc_pos_voltage",
    "dc_pos_current",
    "dc_neg_voltage",
    "dc_neg_current",
    "ac_volts",
    "ac_current",
    "ambient_temp",
    "module_temp_1",
    "module_temp_2",
    "module_temp_3",
    "inveter_temp",
    "das_temperature",
    "dc_power_1",
    "dc_power_2",
    "ac_power_1",
    "ac_power_2",
    "das_battery_voltage",
    "inverter_temp_2",
]


def metric_names() -> dict[int, str]:
    return dict(zip(range(750, 771), NAMES))


def make_day(date, snow=False, poa_ok=False, conflict=False, degr=1.0, seed=0) -> pd.DataFrame:
    """One day of long-format raw data: measured_on, utc_measured_on, metric_id, value."""
    rng = np.random.default_rng(seed)
    date = pd.Timestamp(date)
    idx = pd.date_range(date, periods=1440, freq="1min")
    el = pvlib.solarposition.get_solarposition(idx.tz_localize("Etc/GMT+7"), 39.74, -105.17)[
        "apparent_elevation"
    ].to_numpy()
    s = np.clip(np.sin(np.radians(el)), 0, None)
    cloud = 0.04 if snow else rng.uniform(0.6, 1.0)
    season = 8 * np.cos((date.dayofyear - 200) / 365 * 2 * np.pi)
    hrs = np.asarray(idx.hour) + np.asarray(idx.minute) / 60
    amb = (-6 if snow else 4) + season + 6 * np.sin((hrs - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.1, 1440)
    dc = 6000 * s * cloud * degr + rng.normal(0, 4, 1440)
    mt = amb + 25 * s * cloud
    if poa_ok:
        poa = 800 * s * cloud + rng.normal(0, 10, 1440)
    else:
        poa = -400 - 300 * s + rng.normal(0, 30, 1440)
        poa[el > 0] = np.nan  # logged only at night, like the real sensor
    wide = pd.DataFrame(
        {
            "poa_irradiance": poa,
            "dc_power": dc,
            "ac_power": 0.93 * dc - 2.6,
            "dc_pos_voltage": 200 * (s > 0),
            "dc_pos_current": dc / 400,
            "dc_neg_voltage": 200 * (s > 0),
            "dc_neg_current": dc / 400,
            "ac_volts": 0.0,
            "ac_current": 0.0,
            "ambient_temp": amb,
            "module_temp_1": mt,
            "module_temp_2": mt + 0.3,
            "module_temp_3": mt - 0.2,
            "inveter_temp": 20 + 5 * s,
            "das_temperature": 13.95,
            "dc_power_1": 0.0,
            "dc_power_2": dc / 2,
            "ac_power_1": np.nan,
            "ac_power_2": np.nan,
            "das_battery_voltage": dc / 2,
            "inverter_temp_2": dc / 2,
        },
        index=idx,
    )
    ids = {n: i for i, n in metric_names().items()}
    long = (
        wide.rename(columns=ids)
        .rename_axis("measured_on")
        .reset_index()
        .melt(id_vars="measured_on", var_name="metric_id", value_name="value")
        .dropna()
    )
    long["metric_id"] = long["metric_id"].astype("int32")
    long["utc_measured_on"] = long["measured_on"] + pd.Timedelta("7h")
    long = long[["measured_on", "utc_measured_on", "metric_id", "value"]]
    if conflict:  # two daytime hours get a second, different copy of every reading
        seg = long[long["measured_on"].between(date + pd.Timedelta("10h"), date + pd.Timedelta("12h"))].copy()
        seg["value"] = seg["value"] * 1.2 + 5
        long = pd.concat([long, seg], ignore_index=True)
    return long


def make_profiled(
    cfg, names, start="2021-01-01", end="2021-12-31", step="3D", snow_days=(), conflict_days=()
):
    """Profile a synthetic stretch of days into a daily table (what profile_days returns)."""
    from pvdaq_rul.quality import profile_day

    rows = []
    for d in pd.date_range(start, end, freq=step):
        raw = make_day(d, snow=d in snow_days, conflict=d in conflict_days, seed=int(d.dayofyear))
        rows.append(profile_day(d, raw, names, cfg))
    out = pd.DataFrame(rows)
    out["read_error"] = None
    return out
