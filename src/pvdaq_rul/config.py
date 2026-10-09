# SPDX-License-Identifier: Apache-2.0
"""All thresholds and site settings in one place."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # what to process
    system_id: int = 50

    # site (NREL South Table Mountain, Golden CO) and array geometry
    lat: float = 39.7407
    lon: float = -105.1686
    alt: float = 1795.0
    tilt: float = 40.0  # replace with mount-table value if known
    azimuth: float = 180.0
    sun_elev_day: float = 10.0  # degrees; sun higher than this = daylight

    # signals
    power_channel: str = "dc_power"
    temp_coeff: float = -0.004  # silicon: -0.4 % per degree C above 25 C

    # day-level usability rules
    min_daylight_cov: float = 0.90
    max_daylight_gap_min: int = 60
    max_conflict_share: float = 0.05
    min_clear_minutes: int = 30

    # snow / outage detection
    snow_energy_ratio: float = 0.30
    snow_ambient_c: float = 2.0
    snow_min_run_days: int = 2
    snow_pad_days: int = 1

    # known bad periods: (start, end, reason)
    exclude_periods: tuple = (("2023-02-01", "2023-02-28", "Colorado snowstorm - damaged data"),)
    exclude_years: tuple = ()

    # irradiance sensor validity (day level)
    poa_min_day_cov: float = 0.80
    poa_day_median_range: tuple = (20, 1400)
    poa_night_median_range: tuple = (-20, 30)
    poa_min_corr: float = 0.70

    # year verdict
    use_min_days: int = 150
    use_min_months: int = 9
    caution_min_days: int = 60
