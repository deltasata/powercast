# Lab notebook

Times are Europe/Berlin unless marked UTC. Commands are run from the repository root with
the project environment (`.venv/bin/python`).

## Step 1: Repository setup (22–23 Nov 2026)

- `git init`, branch `main`; local identity `user.name = Satadru Bag`, `user.email` set
  in `.git/config` only.
- Local-only files listed in `.git/info/exclude` (not in `.gitignore`).
- Git hooks: `commit-msg` strips attribution trailers; `pre-push` blocks a push whose
  commits or tracked files contain forbidden words (see the hook for the list).
- Python 3.12 virtual environment in `.venv/` (created with uv 0.13.9; interpreter cached
  in `.cache/`). Packages: numpy 2.5, pandas 3.1, pyarrow 26, scipy 1.18, scikit-learn 1.9,
  statsmodels 0.15, lightgbm 4.7, matplotlib 3.11, requests 2.34, pytest 9.2, ruff 0.17.
  The package `src/powercast` is importable through a `.pth` file in the environment.
- Folders `src/`, `scripts/`, `tests/`, `results/`, `figures/`, `docs/`, `data/`
  (`data/` is in `.gitignore`: it is large and is re-created by the scripts).
- 23 Nov: added `LICENSE` (MIT, Satadru Bag) and created `docs/`. A check of the setup
  found `docs/`, `data/` and `LICENSE` missing; everything else was in place.
- Hardware available: 64 CPU cores, 251 GB RAM, 6 × NVIDIA A10 (23 GB each; two are
  partly used by other jobs).

## Step 2: Download data (23 Nov 2026)

### SMARD

Command: `python scripts/download_smard.py --date 2026-11-23`
(snapshot in `data/raw/smard/2026-11-23/`, 11 series, 5 870 weekly JSON files, ≈ 95 MB).

API structure (no documentation beyond the community description at
github.com/bundesAPI/smard-api): `chart_data/{filter}/{region}/index_{resolution}.json`
lists weekly files; each file holds `[[interval start in ms UTC, value], …]`.

Series identified by probing (23 Nov):

| filter | content | region used | first interval (UTC) |
|---|---|---|---|
| 251 | day-ahead price, DE-AT-LU zone | (ignored) | 2015-01-04 23:00 |
| 4169 | day-ahead price, DE-LU zone | (ignored) | 2018-09-30 22:00 |
| 410 / 411 | load, actual / day-ahead forecast | DE | 2014-12-31 23:00 |
| 4067 / 123 | wind onshore, actual / day-ahead forecast | DE | 2014-12-31 23:00 |
| 1225 / 3791 | wind offshore, actual / day-ahead forecast | DE | 2014-12-31 23:00 |
| 4068 / 125 | solar, actual / day-ahead forecast | DE | 2014-12-31 23:00 |

Observations:

1. Filter 251 is the DE-AT-LU price: identical to Energy-Charts `bzn=DE-AT-LU` for all
   168 hours of the week of 6 Mar 2017 (max |difference| 0.00 €/MWh). Filter 260
   (Slovenia) agreed in the first hours but differed in 35 of 168 hours, so a short
   check is not enough to identify a series.
2. Region `DE-LU` exists for the fundamentals only from 30 Sep 2018; region `LU` has wind
   and solar forecasts only from Mar 2022. Region `DE` covers 2015 onward for every
   series (decision D2.2).
3. **SMARD's quarter-hour price file before 1 Oct 2025 repeats each hourly price four
   times** (all 61 368 hours checked). It carries no 15-minute information there.
4. SMARD fundamentals are energy per interval (MWh per 15 min): mean 2024 load
   13 249 MWh/15 min = 53.0 GW. Converted to MW (× 4) in the processed tables.
5. The fundamentals are genuine 15-minute data back to 2015: no hour has four identical
   quarter-hour values, except solar at night (zeros).

### Energy-Charts

Command: `python scripts/download_energycharts.py --date 2026-11-23` (one JSON per
endpoint and calendar year; 10 s pause between calls because the API answers HTTP 429
to bursts). Endpoints: `price` (DE-AT-LU 2015–2018, DE-LU 2018–2026), `public_power`
(actual generation by type and load), `public_power_forecast` (day-ahead forecasts of
load, solar, wind onshore, wind offshore).

### Fuel prices (gas, CO₂)

Searched 23 Nov 2026 for free daily series:

- Gas (TTF / THE day-ahead or front month): none found with a licence allowing reuse.
  Barchart requires a paid plan for downloads; Investing.com terms forbid storing or
  redistributing its data.
- CO₂ (EU ETS allowances): EEX publishes primary-auction clearing prices as free
  downloads (auctions on 3–4 weekdays). Terms for reuse are unclear. Ember's daily carbon
  price viewer is no longer updated. ICAP gives weekly averages with a 6-month delay.

Decision D2.4: continue without fuel prices.

## Step 3: Check the data (23 Nov 2026)

Command: `python scripts/build_dataset.py --snapshot 2026-11-23`
Outputs `data/processed/2026-11-23/{prices,fundamentals}.parquet` and
`results/data_checks/2026-11-23.json`.

Price target series (EUR/MWh), spliced: DE-AT-LU hourly to 30 Sep 2018, DE-LU hourly
1 Oct 2018 – 30 Sep 2025, DE-LU quarter-hourly from 1 Oct 2025 (delivery days in Berlin
time).

Results:

- Hourly part (2015-01-04 23:00 UTC – 2025-09-30 21:00 UTC): 94 127 values, 0 duplicates,
  0 missing grid points, 0 missing values.
- Quarter-hour part (from 2025-09-30 22:00 UTC): 40 228 grid points, 0 duplicates;
  **20 and 21 Nov 2026 missing in SMARD** (both the hourly and the quarter-hour file,
  still missing on a fresh download at 10:20 on 23 Nov).
- Daylight-saving days: 24 days with 23 or 25 hours (92 or 100 quarter-hours); all 24
  have the right number of intervals.
- After the switch, SMARD's hourly price equals the mean of its four quarter-hour prices
  to within 0.005 €/MWh (rounding) in all 10 009 hours.
- Range −500.00 to +936.28 €/MWh; 4 546 negative intervals, 593 above 500 €/MWh. None
  removed.
- Fundamentals (15-minute grid, 2015-01-01 onward): 0 duplicates, 0 missing grid points.
  Missing days: load forecast 31 Jan 2020 (96 quarter-hours), offshore wind forecast
  4 Jan 2015. Everything else complete up to the publication edge.

### Cross-check SMARD vs Energy-Charts

Command: `python scripts/crosscheck_sources.py --smard 2026-11-23 --ec 2026-11-23`
(`results/data_checks/crosscheck_2026-11-23.json`; differences are SMARD − Energy-Charts,
tolerance 0.01 €/MWh for prices and 1 % of the mean for power).

| Series | Common points | Mean diff | Max abs diff | Share above tol. | Corr. |
|---|---|---|---|---|---|
| Price | 134 163 | 0.00 | 0.00 €/MWh | 0 | 1.00000 |
| Load actual | 417 056 | 0 MW | 98 MW | 0 | 1.00000 |
| Load forecast | 417 020 | 0 MW | 0.5 MW | 0 | 1.00000 |
| Wind onshore actual | 417 056 | +109 MW | 2 620 MW | 0.30 | 0.99965 |
| Wind offshore actual | 417 056 | −0.3 MW | 148 MW | 0.03 | 0.99999 |
| Solar actual | 417 056 | +118 MW | 3 585 MW | 0.33 | 0.99952 |
| Wind onshore forecast | 417 116 | +36 MW | 7 003 MW | 0.03 | 0.99965 |
| Wind offshore forecast | 417 020 | +26 MW | 1 824 MW | 0.07 | 0.99694 |
| Solar forecast | 417 116 | −0.1 MW | 1 016 MW | 0.02 | 1.00000 |

Prices and load agree exactly. Measured wind and solar differ by 1–2 % on average,
consistent with different versions or estimation of the measured values; they are not
model inputs for the target day. Energy-Charts has the prices of 20–21 Nov 2026 that SMARD
lacks; the 192 quarter-hour prices were filled from Energy-Charts (column `source`) by
`python scripts/build_dataset.py --snapshot 2026-11-23 --ec 2026-11-23`. After filling:
0 missing prices, every delivery day has its expected number of MTUs.

### Publication times (no-peeking rules)

`scripts/probe_publication_times.py` polls SMARD and Energy-Charts every 5 minutes on
23 Nov 2026 and logs, per series, how many values for delivery day 24 Nov are published
(`data/raw/publication_probe/2026-11-24/log.csv`). Energy-Charts answers HTTP 404
("no content available") for a day not yet published.

- 10:10: SMARD has the load forecast for 24 Nov (96 quarter-hours). Wind onshore, wind
  offshore and solar forecasts end at 23 Nov 24:00. Energy-Charts has none of them for
  24 Nov.

### Automated no-peeking check

`powercast.nopeek`: each input series has a publication rule giving `available_at` for
every value; the forecast for delivery day D is issued at D−1 11:00. The check
`assert_no_peeking` rebuilds a model's inputs (or the whole forecast) with every value
published after the issue time replaced by random numbers and fails if anything changes.
Tests (`tests/test_nopeek.py`): honest features pass; a model using the target-day price
or the previous afternoon's measured load is caught; issue times on both daylight-saving
days are correct.

## Step 4: Evaluation plan (23 Nov 2026)

Written in `docs/EVALUATION_PLAN.md` before any model was trained. Validation 2019–2020,
test 2021-01-01 – 2026-10-31 with regimes R0–R3, rolling-forward scheme, 99 quantile
levels, CRPS, coverage, PIT, Diebold–Mariano tests, block bootstrap, battery set-up.
One wording fix (M0 in the quarter-hour period) made the same day before any model run,
recorded in the plan's change record.

## Step 5: Naive baseline M0 (23 Nov 2026)

`powercast.models.naive`; validation 2019–2020 (731 days, 17 544 MTUs), choosing the
error window W:

| W (days) | CRPS | MAE | Coverage 50 / 80 / 90 / 98 % |
|---|---|---|---|
| 28 | 8.155 | 10.544 | 46.3 / 74.5 / 83.4 / 91.2 |
| 91 | 7.995 | 10.371 | 49.6 / 77.5 / 87.6 / 95.8 |
| 182 | 8.021 | 10.316 | 51.0 / 78.2 / 88.1 / 96.4 |
| 364 | **7.978** | 10.312 | 51.3 / 80.1 / 89.7 / 97.2 |

Chosen: W = 364 (lowest validation CRPS, as the plan prescribes). The differences
between 91, 182 and 364 days are small (≤ 0.04 €/MWh) and no error bars were computed for
this choice; the longer window also gives the coverage closest to nominal.
No-peeking check passed on 5 validation days, including both daylight-saving days; a
deliberately cheating variant was caught.

### Publication probe, continued (23 Nov 2026, Berlin time)

| Poll | SMARD load fc | SMARD wind / solar fc | Energy-Charts load fc | Energy-Charts wind / solar fc |
|---|---|---|---|---|
| 10:11 | 96 / 96 | 0 | 0 (404) | 0 (404) |
| 10:49 | 96 / 96 | 0 | 0 (404) | 0 (404) |
| 10:55 | 96 / 96 | 0 | **96 / 96** | 0 (404) |
| 11:00 | 96 / 96 | **0** | 96 / 96 | **0** (404) |

At the issue time (11:00) neither source has the wind or solar forecasts for the next day.
The probe keeps running until 20:00 to record when they appear (decision D3.4).

Final polls (probe 11:11–19:57, every 5 min):

| Series for 24 Nov | Energy-Charts | SMARD |
|---|---|---|
| Load forecast | 10:55 (96 / 96) | 10:10 (96 / 96) |
| Wind onshore, offshore, solar forecasts | **18:04** (last empty poll 17:59) | not by 19:57 |
| Price | not probed; present when fetched for scoring at ≈ 13:40 | not by 19:57 |

The 18:00 rule for wind and solar holds. SMARD publishes both the wind/solar forecasts
and the day-ahead price later than Energy-Charts; the daily run therefore fills prices
missing on SMARD from Energy-Charts (already in `scripts/daily_forecast.py`).
The inputs the models do use are on SMARD in time: the snapshot downloaded at
10:13–10:16 on 23 Nov already holds the wind and solar forecasts up to 23 Nov 23:45
(D−1 for delivery 24 Nov) and the load forecast up to 24 Nov 23:45, so the live run sees
the same information set as the backtest.

## Environment additions (23 Nov 2026)

- uv 0.13.9 binary downloaded from the official GitHub release into `.cache/bin/`
  (SHA-256 verified against the release checksum), because the environment has no pip.
- PyTorch: the default wheel (2.15.0, CUDA 13.2) cannot use the installed NVIDIA driver
  525.60.13 (CUDA 12.0). Replaced by `torch==2.14.1+cu126` from the PyTorch index (CUDA
  12.x minor-version compatibility). Command:
  `UV_CACHE_DIR=.cache/uv .cache/bin/uv pip install --python .venv/bin/python
  --index https://download.pytorch.org/whl/cu126 "torch==2.14.1+cu126"`.
- `chronos-forecasting==2.3.2` (Apache-2.0) with torch pinned; Chronos-2 weights
  (478 MB) in `.cache/hf/` (`HF_HOME`).
- GPUs: from about 10:20 all six A10s were in use by nnUNet training jobs of another
  project on the same account (63–99 % utilisation); new CUDA contexts failed with "busy
  or unavailable". Those jobs also use many CPU cores, so from 11:25 this project limits
  itself to about 20 parallel worker processes, to avoid slowing them down. Adam's optimizer step probes CUDA
  whenever it is visible, so CPU runs set `CUDA_VISIBLE_DEVICES=""`. The network (M3)
  trains in 20–30 s per fit on one CPU thread; Chronos-2 runs at ≈ 8 s per day on 4 CPU
  threads, so all models fit the CPU budget.

## Step 6: LEAR-QRA (M1), validation (23 Nov 2026)

- `python scripts/run_lear.py --snapshot 2026-11-23 --start 2018-01-01 --end 2020-12-31`
  (1 096 days, 60 processes, 402 s). Exogenous lags from the publication rules: load
  forecast D, D−1; wind + solar forecast D−1, D−2. 199 inputs per day.
- scikit-learn ≥ 1.1 cannot estimate the noise variance in `LassoLarsIC` when samples <
  features (56- and 84-day windows, 199 inputs). The noise variance is set to Var(y), as
  scikit-learn < 1.1 did inside the epftoolbox reference implementation.
- No-peeking check on the full LEAR forecast function: passed on 5 days (both
  daylight-saving days of 2018, 5 Jun 2019, and 31 Jan / 1 Feb 2020 around the missing
  load-forecast day).
- Validation MAE of the LEAR point forecasts (hourly slots, 2019–2020): window 56 days
  6.609, 84 days 6.543, 364 days 6.259, 728 days 6.070, mean of the four 6.124 €/MWh
  (naive: 10.31).

- QRA on the four LEAR point forecasts (`scripts/run_qra.py`, statsmodels QuantReg,
  one regression per hour and level, 20 processes), validation 2019–2020:

  | QRA window (days) | CRPS | MAE | Coverage 50 / 80 / 90 / 98 % |
  |---|---|---|---|
  | 91 | 4.761 | 6.258 | 46.3 / 74.5 / 84.6 / 92.1 |
  | 182 | 4.698 | 6.172 | 48.8 / 77.3 / 87.5 / 95.8 |
  | 364 | **4.643** | 6.130 | 49.6 / 79.3 / 88.9 / 96.8 |

  Chosen: 364 days (D6.3). CRPS 42 % below the naive M0 (7.978) on the same days; error
  bars and the Diebold–Mariano test come with the test-period evaluation (step 10).
  Runtime ≈ 52 s per forecast day on one core.

## Step 12 (draft): daily run, first live test (23 Nov 2026)

- `scripts/daily_forecast.py`: refreshes the processed tables (last three weekly SMARD
  files; Energy-Charts for missing prices), applies the publication rules for the
  issue time, forecasts with the models in `config/live.json`, applies the quarter-hour
  layer, writes `forecasts/live/<day>.csv/.npz` and scores earlier live forecasts.
- `.github/workflows/daily.yml`: runs at 08:40 and 09:40 UTC (10:40 Berlin in summer and
  winter); caches `data/live/` between runs; commits `forecasts/live/`.
  `requirements-live.txt` pins the package versions of this environment.
- Test at 11:41 on 23 Nov with M0 only (state seeded from snapshot 2026-11-23):
  forecast for 24 Nov, 96 quarter-hours, information set "published by 10:00 UTC",
  not late. Naive median at 00:00 = 220.6 €/MWh (price one week earlier 208.6 €/MWh;
  prices in mid-November 2026 between ≈ 90 and 320 €/MWh).
- First live score (24 Nov 2026, forecast issued 23 Nov 11:42 from the 11:00 information
  set, M0 only): CRPS 31.6 €/MWh, MAE 42.5 €/MWh, 80 % coverage 72 % and 90 % coverage
  92 % over 96 quarter-hours. Real prices averaged 254 €/MWh against a forecast median
  of 213 €/MWh. One day: a check that the pipeline works, not a measurement of skill.
  Copied to `forecasts/live/` as the first track-record entry.

## Steps 10–11: tooling, tried on validation (23 Nov 2026)

- `powercast.guard`: the no-peeking check on the exact inputs of one refit (training
  features, training targets, target-day features), now run by `run_model.py`,
  `tune_gbt.py` and `tune_ddnn.py` for every training window used.
- `scripts/run_battery.py`: battery LP per day on the forecast mean price of each MTU,
  settled at real prices; profit per MW per year with block-bootstrap intervals, paired
  difference to M0 and a one-sided Diebold–Mariano test on daily profits.
  Validation 2019–2020 (a check of the tool, not a result):

  | Schedule from | Profit (EUR/MW/yr) | 95 % interval | Share of perfect |
  |---|---|---|---|
  | perfect foresight | 17 984 | 16 734 – 19 089 | 100 % |
  | M0 naive | 13 673 | 12 679 – 14 621 | 76.0 % |
  | M1 LEAR-QRA | 15 057 | 13 933 – 16 096 | 83.7 % |

  M1 − M0 = +1 384 EUR/MW/yr (95 % interval 933 – 1 873; DM p = 3 × 10⁻⁹).
- `scripts/evaluate_models.py --period validation` with M0 and M1: CRPS 7.978
  [7.101, 8.623] and 4.643 [4.142, 4.951] €/MWh (14-day block bootstrap).
- `scripts/make_figures.py`: CRPS by market period, coverage, PIT, CRPS by hour,
  battery; validation versions in `figures/validation_*.png`. The colour palette is a
  fixed categorical order; the palette validator needs Node.js, which is not installed,
  so the palette was taken from a published, already-validated set without re-running it.

## Step 7: LightGBM (M2), tuning (23 Nov 2026)

- `python scripts/tune_gbt.py --snapshot 2026-11-23 --n 30 --parallel 4 --threads 5`
  (13:33–16:58, 20 threads). No-peeking check passed for all three training windows.
- Mean pinball loss over levels 0.05/0.25/0.5/0.75/0.95, refit every 28 days: best
  2.137, worst 2.504 €/MWh. Longer windows win clearly (best per window: 364 days 2.318,
  728 days 2.200, 1 092 days 2.137). Chosen configuration in D7.3.
- Full validation run with the chosen configuration (refit every 7 days, 23 levels):
  `python scripts/run_model.py --model gbt --config results/tuning/gbt_final.json
  --snapshot 2026-11-23 --start 2019-01-01 --end 2020-12-31 --jobs 4 --threads 5`
  (19:50–20:24). CRPS 4.689, MAE 6.250 €/MWh; coverage 42.6 / 69.1 / 79.8 / 93.2 % at
  50 / 80 / 90 / 98 %. Accuracy close to M1 (4.643), but the intervals are too narrow.
  A possible cause (not tested here): the quantile models are fitted on in-sample
  residuals that the trees partly memorise, which pulls the outer quantiles inwards. The
  plan has no recalibration step, so M2 is carried forward unchanged.
- The remaining runs (network tuning, full validation runs of M2–M4, test forecasts of
  M0 and M1 with scores withheld) run as one queue on the 20-worker budget; logs in
  `logs/` (not in git).

## Step 8: distributional network (M3), tuning (23 Nov 2026)

- `CUDA_VISIBLE_DEVICES="" python scripts/tune_ddnn.py --snapshot 2026-11-23 --dist
  {normal,jsu} --n 30 --jobs 20` (17:04–19:50, both distributions).
- Validation CRPS of the best configuration (one seed, refit every 28 days): normal
  4.432, JSU 4.405 €/MWh (M1 LEAR-QRA, refit daily: 4.643). Coverage 80 % of the best:
  normal 81.7 %, JSU 78.1 %. Three of 60 configurations diverged (large learning rates).
  Chosen configurations in D8.2.
- Full validation runs (refit every 7 days, 5 seeds, 20 processes):
  `python scripts/run_model.py --model ddnn --config results/tuning/ddnn_normal_final.json
  --snapshot 2026-11-23 --start 2019-01-01 --end 2020-12-31 --seeds 0 1 2 3 4 --jobs 20`
  (normal: 20:24–21:15; JSU: 21:15–22:00). Seed ensemble = mean of the 5 seeds' quantiles
  (`scripts/ensemble_seeds.py`).

  | Output | CRPS per seed, mean ± sd | Ensemble CRPS | Ensemble MAE | Ensemble coverage 50 / 80 / 90 / 98 % |
  |---|---|---|---|---|
  | normal | 4.405 ± 0.018 | 4.308 | 5.836 | 53.4 / 82.5 / 91.0 / 97.6 |
  | JSU | 4.370 ± 0.016 | 4.325 | 5.854 | 48.5 / 78.7 / 89.6 / 98.2 |

  Averaging 5 seeds lowers the CRPS by 0.10 (normal) and 0.05 (JSU) €/MWh, 3–6 times the
  seed-to-seed spread. The two outputs are within 0.02 €/MWh of each other; the plan
  reports both.

## Step 9: Chronos-2 (M4), validation (23 Nov 2026)

- `HF_HUB_OFFLINE=1 python scripts/run_model.py --model chronos --variant
  {univariate,covariates} --snapshot 2026-11-23 --start 2019-01-01 --end 2020-12-31
  --jobs 5 --threads 4` (zero-shot, CPU, context 8 192 hours = 341 days).
  No-peeking check passed on the exact model inputs (3 days).
- Prices only: CRPS 4.552, MAE 6.106 €/MWh, coverage 44.8 / 75.4 / 87.4 / 97.4 %
  (4 min). With covariates: CRPS 4.419, MAE 5.940, coverage 46.5 / 77.3 / 88.5 / 97.9 %
  (11 min). Chosen: covariates (D9.2).
- Without any training on this market, Chronos-2 with prices only already scores better
  on validation than LEAR-QRA (4.643), which is fitted daily with load and wind/solar
  forecasts. Possible memorisation of the price history during pretraining is recorded
  as a limitation in D9.2.

## Validation summary, all models (23 Nov 2026)

`python scripts/evaluate_models.py --period validation` with M0, M1, M2, both M3 seed
ensembles and M4 (covariates); full tables from `python scripts/report.py --period
validation`. CRPS [95 % block-bootstrap interval], €/MWh:

| Model | CRPS | MAE | Coverage 50 / 80 / 90 / 98 % |
|---|---|---|---|
| M0 naive | 7.978 [7.101, 8.623] | 10.312 | 51.3 / 80.1 / 89.7 / 97.2 |
| M1 LEAR-QRA | 4.643 [4.142, 4.951] | 6.130 | 49.6 / 79.3 / 88.9 / 96.8 |
| M2 LightGBM | 4.689 [4.192, 5.005] | 6.250 | 42.6 / 69.1 / 79.8 / 93.2 |
| M3 normal, 5-seed ensemble | 4.308 [3.851, 4.586] | 5.836 | 53.4 / 82.5 / 91.0 / 97.6 |
| M3 JSU, 5-seed ensemble | 4.325 [3.855, 4.608] | 5.854 | 48.5 / 78.7 / 89.6 / 98.2 |
| M4 Chronos-2 (covariates) | 4.419 [3.877, 4.771] | 5.940 | 46.5 / 77.3 / 88.5 / 97.9 |

- The intervals overlap because every model shares the same hard days; the paired
  Diebold–Mariano test is sharper: M3 (both) and M4 beat M1 and M2 (p ≤ 0.003); M1 vs
  M2, and the three complex models among themselves, are not distinguishable (p ≥ 0.09).
- 80 % coverage on price spikes: M2 9 %, M1 24 %, M3/M4 46–48 %; on negative prices
  M1 40 %, M3/M4 53–63 %. All models are overconfident in these tails.
- Settings are now frozen for the test period: M0 W = 364; M1 QRA window 364; M2 D7.3;
  M3 D8.2; M4 D9.2.

## Step 12: daily run, fixes before the first M1 live run (24 Nov 2026)

- A forced dry run at 08:20 failed: SMARD had neither the 24 Nov prices nor the 24 Nov
  wind and solar forecasts yet (both are D−1 inputs for delivery 25 Nov), so the LEAR
  inputs contained missing values. The backtest assumes them known from D−2 13:00
  (price) and D−2 18:00 (forecasts), which holds on Energy-Charts (probe of 23 Nov) but
  not on SMARD.
- Fix (`build.extend_from_ec`, `daily_forecast.fetch_ec`): each run downloads the last
  three weeks of Energy-Charts prices and day-ahead forecasts and uses them to fill and
  extend the newest SMARD values, for that run only (the stored tables stay SMARD).
  Justified by the cross-check: prices identical; forecast mean absolute differences
  0.15 (load), 36 (wind onshore), 30 (wind offshore), 4 (solar) MW.
- `--day` option for test and backfill runs (marked `backfill` in the output and the
  track record); the "too early" guard now compares with 10:15 on D−1; `late` means
  issued after 12:00 on D−1.
- Past LEAR forecasts for the first run come from the backtest
  (`config/seed/lear_points.parquet`, Oct 2025 – Oct 2026, 396 days, 0.4 MB); missing
  days are computed in parallel.
- Dry run for 24 Nov (backfill, M0 + M1): 2.5 min on 8 cores; filled from
  Energy-Charts: 96 prices, 96 values of each wind/solar forecast.
- Quantiles of every live forecast are clipped to the price limits (D10.1).

## Steps 10–11: test period, run once (24 Nov 2026)

- Test forecasts with frozen settings (queues of 23–24 Nov, 20 workers): M0 (22:16),
  LEAR points 2021–2026 (22 min) and QRA (2 h 40 min), M4 Chronos-2 (25 min), M2 LightGBM
  (1 h 36 min), M3 normal and JSU, 5 seeds each (2 h 41 min and 2 h 11 min). No-peeking
  check passed before every run (5 check days, including the gas crisis and the first
  25-hour quarter-hour day). Scores were withheld by the run scripts.
- Before scoring, a finiteness check found infinite quantiles in M3; fixed by clipping to
  the auction's price limits for every model (D10.1, plan change record). Share of
  clipped quantiles: M3-JSU 0.11 % (233 days), M3-normal 0.013 % (33 days), others
  ≤ 0.003 % (`results/evaluation/clipped_share_test.json`).
- `python scripts/evaluate_models.py --period test ...` and
  `python scripts/run_battery.py --period test ...` (08:16–09:00), then
  `scripts/make_figures.py --period test` and `scripts/report.py --period test`
  (`results/report_test.md`). Main numbers (CRPS, €/MWh, 95 % block-bootstrap interval):

  | Model | CRPS | MAE | Coverage 80 % | Battery, EUR/MW/yr above M0 |
  |---|---|---|---|---|
  | M0 naive | 33.93 [30.08, 37.62] | 43.46 | 75.3 % | – |
  | M1 LEAR-QRA | 17.49 [15.91, 19.00] | 23.26 | 76.6 % | +4 722 [3 894, 5 470] |
  | M2 LightGBM | 19.44 [17.32, 21.60] | 25.69 | 66.2 % | +2 429 [1 161, 3 474] |
  | M3 normal (ensemble) | 17.43 [15.61, 19.24] | 23.64 | 82.9 % | +5 296 [4 381, 6 167] |
  | M3 JSU (ensemble) | 17.42 [15.60, 19.20] | 23.36 | 80.2 % | +5 894 [4 998, 6 698] |
  | M4 Chronos-2 | **15.42** [14.05, 16.75] | 20.99 | 78.3 % | **+6 725** [5 873, 7 561] |

- Diebold–Mariano: M4 beats every other model on the whole period (p < 0.001) and in
  every market period (p ≤ 0.04). M3 does **not** beat M1 on the test period (p = 0.40
  normal, 0.37 JSU), although it did on validation: it wins after the crisis (R2,
  p < 0.001) and loses during the gas crisis (R1, M1 better than M3-JSU, p = 0.03).
  M2 is the weakest learned model and clearly overconfident (80 % interval: 66 %).
- Battery: perfect foresight 75 547 EUR/MW/yr; M0 reaches 81.7 % of it, M4 90.6 %.
  The profit order follows the CRPS order. Paired differences: M4 − M1 = +2 003
  [1 448, 2 643] EUR/MW/yr (p = 2 × 10⁻¹²); M4 − M3-JSU = +831 [339, 1 409] (p = 6 × 10⁻⁴).
- Memorisation check for M4 (`results/evaluation/robustness_test.json`): the weights
  used are identical to the first release of 30 Oct 2025; on 1 Nov 2025 – 31 Oct 2026 M4
  still beats M1 (16.07 vs 16.97, p = 0.003).
- Live models (D10.2): M4 published, M0 and M1 as references; `config/live.json`.
  Chronos-2 revision pinned in `powercast.models.chronos2.REVISION`; weights default to
  `.cache/hf` inside the project. `requirements-deep.txt` (CPU PyTorch) and a weights
  cache added to the workflow.
- Live script vs backtest on 31 Oct 2026 (backfill): medians agree within 1.6 €/MWh (M4)
  and 4.4 €/MWh (M1); the residual comes from the quarter-hour layer's Monte Carlo and
  data vintage.

## Live forecast for 25 Nov 2026 (24 Nov)

- The scheduled run at 10:20 failed: the load forecast for 25 Nov was on neither SMARD
  nor Energy-Charts (checked again at 10:23). On 23 Nov SMARD had it at 10:10, so its
  publication time varies; the rule "public at D−1 10:00" (D3.4) is the regulatory
  deadline, not a guarantee.
- Fix (`daily_forecast.wait_for_load`): the run polls every 5 minutes until the load
  forecast for D is complete, up to 11:40; if it is still missing then, the plan's rule
  for missing inputs applies (value one week earlier) and the forecast is flagged
  `load_forecast_from_last_week`. The output now records both the start time and the
  time the forecast was made (`started_berlin`, `issued_berlin`).
- Re-run started 10:24: load forecast missing at 10:25, present on Energy-Charts at the
  10:30 poll (still missing on SMARD); forecast written at 10:39 with M0, M1 and M4,
  not late. Its `issued_berlin` field still shows the start time 10:24:58 (the field
  was corrected afterwards for later runs).
- Backtest note: the backtest uses the load forecast for D as known at the issue time
  (11:00), consistent with both days observed (published by 10:10 and by 10:30).
- `requirements-deep.txt` pins the CPU build `torch==2.14.1+cpu` (checked on the PyTorch
  index for Python 3.12, Linux x86_64). When today's forecast already exists, a run
  still refreshes the stored data, so a manual Actions run warms the data cache.


## First GitHub Actions runs (24–25 Nov 2026)

- Two scheduled runs on 24 Nov (started 20:41 and 21:20 Berlin, not at the 10:40
  schedule: GitHub's scheduler delays or shifts runs) both failed in the Forecast step
  within one second. The job's public step list and annotations show only
  "exit code 1"; the full log needs a signed-in account.
- Cause: `.gitignore` had the rule `data/`, which matches every folder named `data`,
  including the code package `src/powercast/data/` (download, build, update, checks,
  Energy-Charts reader). The six files were never committed, so the import failed on
  GitHub, while every local run worked. Found with
  `git ls-files -o -i --exclude-standard`; nothing else was missing (apart from the
  files excluded on purpose).
- Fix (commit 850d870, 25 Nov 09:37): the rule is now `/data/` (top-level data folder
  only) and the package is committed. Trace scan clean; 23 tests pass.
- Check: a fresh `git clone` of the fixed commit, run with an empty data folder as on
  GitHub (`daily_forecast.py --day 2026-11-25`, refresh path), downloaded the full
  SMARD history (6 034 files) and finished correctly in 212 s.
- Side effect found: until this commit the public repository could not be run by
  anyone cloning it.
- Warnings (harmless for now): the actions used (`checkout@v4`, `setup-python@v5`,
  `cache@v4`) target Node.js 20, which GitHub has deprecated; "Cache save failed" came
  from the empty data folder of the failed runs.

## Live forecast for 26 Nov 2026 and the new schedule (25 Nov)

- No GitHub run started for either scheduled slot of 25 Nov (watched via the public API
  every 90 s until 11:20 Berlin). Fallback: the forecast for 26 Nov was made on this
  machine (`python scripts/daily_forecast.py`), started 11:21, written 11:24, not late;
  load forecast for D present on SMARD/Energy-Charts. Log: `logs/live_2026-11-26.log`
  (not committed).
- First live scores (track record, one day each, so no error bars yet): 25 Nov CRPS
  M4 9.88, M1 11.63, M0 73.31 €/MWh.
- Schedule redesigned and actions upgraded (D12.1, D12.2):
  ten triggers through the night and morning, a wait step until 10:15 Berlin, checkout
  of the newest `main` after the wait, publish with pull-rebase; `checkout@v7`,
  `setup-python@v7`, `cache@v6`. Forecasts now record `runner` (`github-actions` or
  `local`); the three forecasts so far (24–26 Nov) were all made locally.

## Final audit of the public repository (25 Nov 2026)

- Independent read-only audit of README, docs and saved results. Every number in the
  README matched the saved results; no traces found; all linked files tracked. Fixed:
  - Reproduce section: it did not say how M1 gets the 2020 LEAR forecasts its first test
    window needs, and left out the M2/M3 forecast and seed-ensemble commands.
    `run_qra.py --points` now takes several files and joins them; the file used in the
    test run (`lear_points/2026-11-23_2018-01-01_2026-10-31.parquet`) is exactly the
    validation file followed by the test file (checked, identical). Commands added.
  - `results/evaluation/robustness_test.json` had been written by a one-off command; it is
    now made by `scripts/robustness.py --snapshot 2026-11-23` (reads saved forecasts and
    scores only, nothing refitted; 2 min). All earlier values reproduced exactly; it adds
    the CRPS with intervals after the Chronos-2 release and the seed spread of M3, which
    the plan (section 4) promised and which had not been computed for the test period.
  - Seed spread on the test period (`scripts/report.py`, new table): single seeds CRPS
    17.99 ± 0.12 (normal) and 17.61 ± 0.09 (JSU) €/MWh, mean ± sd over 5 seeds; seed
    ensembles 17.43 and 17.42. A single seed scores worse than M1 (17.49); the averaging
    over seeds is what brings the network level with LEAR.
  - The M3 p-values in the step 10 entry above were swapped (normal 0.40, JSU 0.37);
    corrected. README: issue-time wording (information up to 11:00, job starts 10:15),
    live forecasts so far made by the local fallback, abbreviations defined, PIT reading
    of M3-normal (hump near 0.7: forecasts too low), p = 2.5 × 10⁻¹², pointers to the
    saved results not shown in the README.
