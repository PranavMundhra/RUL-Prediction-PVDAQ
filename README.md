# pvdaq-rul

Find out **which part of the PVDAQ System 50 data can be used to train a Remaining Useful Life (RUL)
model for solar panels**, and write that part in a tidy format for the next notebooks.

> **Status:** early. The code and tests pass on synthetic System 50 data. The pipeline notebook has
> not yet been run against the real OEDI bucket, so no real-data results are published here.
> No model is trained in this repository yet.

## Why this exists

Raw PVDAQ System 50 data cannot go straight into a model. Earlier checks found:

- the same sensor logged more than once for the same minute, sometimes with **different values**;
- an irradiance sensor (`poa_irradiance`) that is missing about half the time and often unphysical;
- damaged periods, such as the February 2023 Colorado snowstorm.

So this project scores **every day**, then every year, and keeps only trustworthy days.

## How the data is handled

| Problem | What the pipeline does |
|---|---|
| Identical repeated readings | keeps one copy |
| Conflicting repeated readings | sets them to missing, **never averages** them |
| Unreliable POA sensor | not used; health indicator comes from a clear-sky model (power / modelled irradiance, temperature corrected) |
| Snow, outages, known bad periods | detected automatically and/or excluded by hand in the settings |
| Missing days | listed explicitly as `no_data`, never filled |

Each day gets exactly one verdict (`usable` or the first rule it failed), and each year is rated
`USE`, `USE_WITH_CAUTION` or `DO_NOT_USE`.

## Quick start

```bash
git clone https://github.com/<your-github-username>/pvdaq-rul.git
cd pvdaq-rul
make install        # pip install -e ".[dev]" and pre-commit hooks
make test           # synthetic data only, no internet needed
```

Run the full analysis (needs internet, tens of minutes for all years):
open `notebooks/01_data_usability.ipynb` (works on Kaggle too). Every code cell has a plain-English
markdown cell before it. Set `DAY_STRIDE = 7` for a quick trial.

Or use the library:

```python
from pvdaq_rul import Config
from pvdaq_rul.io import make_fs, load_metric_names, build_inventory, group_days
from pvdaq_rul.quality import profile_days
from pvdaq_rul.usability import build_daily, flag_snow, assign_reasons, year_scorecard

cfg = Config()                       # all thresholds live here
fs = make_fs()
names = load_metric_names(fs, cfg.system_id)
days = group_days(build_inventory(fs, cfg.system_id))
profiled = profile_days(days, fs, names, cfg)

daily = assign_reasons(flag_snow(build_daily(profiled), cfg), cfg)
print(year_scorecard(daily, cfg))
```

## Layout

```text
src/pvdaq_rul/      io, quality (day profiler), health_index, usability (verdicts), config
tests/              unit + end-to-end tests on synthetic data (tests/fake_pvdaq.py)
notebooks/          01_data_usability.ipynb  (standalone, Kaggle-friendly)
ml_pipeline/        PIPELINE.md progress checklist, gate approvals, figures
data/               git-ignored; nothing is downloaded into git
```

The notebook and the package contain the same logic; the notebook is self-contained so it runs on
Kaggle without installing this package.

## Pipeline rules

Work follows a 16-step ML pipeline with approval gates (see `ml_pipeline/PIPELINE.md`).
Outputs that feed RUL training (`rul_ready_daily.parquet`) stay locked until Gate A, the approval of
the prediction-problem definition, is recorded.

## Licenses

| What | License |
|---|---|
| Source code | Apache-2.0 (`LICENSE`, `NOTICE`) |
| Documentation and figures | CC BY 4.0 (`LICENSE-DOCS.md`) |
| PVDAQ data (not included) | Creative Commons Attribution, see `DATA_LICENSE_NOTICE.md` |

## Citing

Please cite this software (`CITATION.cff`) **and** the dataset:

> NREL. (2021). *Photovoltaic Data Acquisition (PVDAQ) Public Datasets* [data set].
> https://dx.doi.org/10.25984/1846021

NREL, DOE and OEDI do not endorse this project.
