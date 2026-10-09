# SPDX-License-Identifier: Apache-2.0
"""Clear-sky performance index: a health indicator that does not need the POA sensor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from .config import Config


def clearsky_poa(info: pd.DataFrame, idx_utc: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    """Modelled clear-sky plane-of-array irradiance (W/m2) for each minute."""
    loc = pvlib.location.Location(cfg.lat, cfg.lon, tz="UTC", altitude=cfg.alt)
    cs = loc.get_clearsky(idx_utc, model="ineichen")
    poa = pvlib.irradiance.get_total_irradiance(
        cfg.tilt,
        cfg.azimuth,
        info["zen"].to_numpy(),
        info["az"].to_numpy(),
        cs["dni"].to_numpy(),
        cs["ghi"].to_numpy(),
        cs["dhi"].to_numpy(),
    )
    return pd.Series(np.asarray(poa["poa_global"]), index=info.index)


def performance_index(
    power: pd.Series, module_temp: pd.Series, info: pd.DataFrame, poa_cs: pd.Series, cfg: Config
) -> dict:
    """Median power / clear-sky POA on stable, near-clear minutes (raw and temperature-corrected)."""
    out = {"clear_minutes": 0, "perf_index_raw": np.nan, "perf_index_tc": np.nan}
    ok = (info["elev"] > 15) & (poa_cs > 300) & (power > 0)
    r = (power / poa_cs)[ok]
    if len(r) < 20:
        return out
    cv = r.rolling("15min").std() / r.rolling("15min").mean()
    clear = (cv < 0.05) & (r > 0.8 * r.quantile(0.9))
    out["clear_minutes"] = int(clear.sum())
    if clear.sum() >= 5:
        tc = 1 + cfg.temp_coeff * (module_temp[ok] - 25.0)
        out["perf_index_raw"] = float(r[clear].median())
        if tc[clear].notna().any():
            out["perf_index_tc"] = float((r / tc)[clear].median())
    return out
