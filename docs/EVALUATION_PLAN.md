# Evaluation plan

Written on 23 Nov 2026, before any model was trained or any forecast scored. Only data
checks (counts, gaps, ranges) had been run on the data. Changes after this date are
listed in the change record at the end, with the reason and the date.

## 1. Forecasting task

- **Delivery day D** (Europe/Berlin calendar day), bidding zone DE-LU (DE-AT-LU before
  1 Oct 2018).
- **Target:** the day-ahead price of every market time unit (MTU) of day D, in EUR/MWh:
  24 hourly prices before 1 Oct 2025 (23 or 25 on daylight-saving days), 96
  quarter-hourly prices from 1 Oct 2025 (92 or 100).
- **Issue time:** the forecast for D is issued at a fixed clock time on D−1, before the
  12:00 gate closure (`powercast.nopeek.ISSUE_TIME_LOCAL`, see D3.4). It may use only
  values published by then. Every model passes `powercast.nopeek.assert_no_peeking`
  before every run.
- **Forecast object:** the predictive distribution of each MTU price, represented by its
  quantiles at the 99 levels τ = 0.01, 0.02, …, 0.99. Quantiles are sorted after
  prediction, so they never cross.

## 2. Periods (delivery days, inclusive)

| Period | Dates | Use |
|---|---|---|
| History | 2015-01-05 – 2018-12-31 | training only |
| Validation | 2019-01-01 – 2020-12-31 (731 days) | choosing hyperparameters, windows, model variants |
| Test | 2021-01-01 – 2026-10-31 (2 130 days) | final scores, run once with frozen settings |
| Live | from the first daily run onward | track record of the daily job |

Test sub-periods (market regimes), fixed now:

| Regime | Dates | MTU |
|---|---|---|
| R0 pre-crisis | 2021-01-01 – 2021-08-31 | 60 min |
| R1 gas crisis | 2021-09-01 – 2022-12-31 | 60 min |
| R2 post-crisis | 2023-01-01 – 2025-09-30 | 60 min |
| R3 quarter-hour | 2025-10-01 – 2026-10-31 | 15 min |

The test period is run once, after all settings are frozen on the validation period.
No setting is changed after test scores are seen. If a run fails for a technical reason
(crash, bug that makes the run invalid), the fix and the re-run are recorded below.

## 3. Rolling-forward scheme

For each delivery day D in the validation and test periods, in date order:

1. Build the information set at issue time for D (`nopeek.mask_unpublished`).
2. If the model is due for refitting, refit it on the training window ending with the
   last delivery day whose price is known (D−1).
3. Forecast all MTUs of D.

Refitting frequency (compute budget, fixed now):

| Model | Refit | Training window |
|---|---|---|
| Naive | daily | error window W chosen on validation from {28, 91, 182, 364} days |
| LEAR-QRA | daily | LEAR windows {56, 84, 364, 728} days; QRA window chosen on validation from {91, 182, 364} days |
| LightGBM | every 7 days | chosen on validation from {364, 728, 1092} days |
| Distributional NN | every 7 days | chosen on validation from {728, 1092} days |
| Foundation model | never (zero-shot) | context = most recent prices up to the model's context limit |

Settings chosen on validation are frozen for the test period.

### Daylight-saving days
Models work on 24 local-hour slots. Training targets: on 25-hour days the two 02:00
prices are averaged; on 23-hour days the missing 02:00 slot is the mean of 01:00 and
03:00. Forecasts: on 25-hour days both 02:00 hours receive the slot-02 forecast; on
23-hour days slot 02 is dropped. Scores use only MTUs that exist.

### Quarter-hour prices (from 1 Oct 2025)
All models forecast the **hourly mean price** P_h (before the switch, the hourly price;
after it, the mean of its four quarter-hour prices). The quarter-hour price is
P_{h,q} = P_h + δ_{h,q}, with the intra-hour deviation δ predicted by one common,
model-independent layer:

- δ̂_{h,q} = linear regression of δ on the within-hour deviation of the quarter-hour
  residual-load forecast (load − wind − solar forecasts) and on quarter-of-hour ×
  hour-of-day dummies, fitted on quarter-hour days known at issue time;
- predictive distribution of P_{h,q}: Monte Carlo sum of draws from the model's P_h
  quantiles and draws from the δ-layer's past residuals (assumed independent), 2 000
  draws, then the 99 quantiles;
- until 14 quarter-hour days are known (deliveries 1–14 Oct 2025), δ̂ = 0 and no
  residual is added.

Because the layer is shared, differences between models in R3 come from their hourly
forecasts. R3 is also scored on hourly means, to compare with R0–R2.

### Missing inputs
A missing input value is replaced by the value at the same time one week earlier
(available at issue time). Known cases: load forecast on 31 Jan 2020; offshore wind
forecast on 4 Jan 2015.

## 4. Models

| ID | Model | Class |
|---|---|---|
| M0 | Naive: price of the same local hour one week earlier (hourly mean from Oct 2025, then the common quarter-hour layer), plus empirical quantiles of this rule's past errors at the same local hour over the last W days | simple |
| M1 | LEAR-QRA: LEAR point forecasts (Lago et al. 2021) for 4 calibration windows, combined by quantile regression averaging (Nowotarski & Weron 2015) per hour and τ | simple |
| M2 | LightGBM with quantile loss | simple |
| M3 | Distributional neural network (Marcjasz et al. 2023), normal and Johnson's SU outputs, 5 random seeds each | complex |
| M4 | Pretrained time-series foundation model, zero-shot (Chronos-2; fallback Chronos-Bolt or TimesFM if it does not run on one GPU) | complex |

Shared inputs (M1–M3), all subject to the no-peeking check: prices of D−1, D−2, D−3 and
D−7; day-ahead forecast of load for D and D−1; day-ahead forecasts of wind onshore +
offshore and solar for the latest two days published at issue time (D−1 and D−2, see
D3.4); day-of-week. LightGBM also gets the hour of day and month. Prices are modelled after the
variance-stabilising transform asinh((p − median)/MAD), median and MAD (median absolute
deviation) computed on the training window (Uniejewski et al. 2018). Quantiles are
equivariant under monotone transforms, so they are back-transformed exactly.

Details fixed now:

- **M1** LEAR: one LASSO per hour on the transformed price; regularisation strength by
  LARS with AIC, as in the epftoolbox reference implementation. QRA: linear quantile
  regression of the price on the 4 LEAR forecasts, for each hour and each τ.
- **M2** quantile levels {0.01, 0.02, 0.05, 0.10, 0.15, …, 0.90, 0.95, 0.98, 0.99}
  (23 models); other levels by linear interpolation. Hyperparameters (number of leaves,
  learning rate, number of trees, minimum data per leaf) by random search of at most 30
  configurations on the validation period.
- **M3** multilayer perceptron with 2 hidden layers predicting the 24 hourly
  distributions' parameters; loss = negative log-likelihood. Architecture, dropout,
  learning rate and regularisation by random search of at most 30 configurations on the
  validation period, separately for the normal and the JSU output. Reported per seed
  (mean and spread over seeds) and as the seed ensemble (average of the 5 seeds'
  quantiles).
- **M4** native output quantiles linearly interpolated to the 99 levels; levels outside
  the model's range take its most extreme quantile (stated as a limitation).

**Rule for complex models:** M3 or M4 is kept for the daily job only if it beats the best
of M0–M2 on the whole test period (lower CRPS and one-sided Diebold–Mariano p < 0.05).

## 5. Scores

For a quantile forecast q_τ and price y, the pinball loss is
ρ_τ(y, q_τ) = (𝟙{y < q_τ} − τ)(q_τ − y).

- **CRPS** (continuous ranked probability score), primary score, in EUR/MWh:
  CRPS = (2/99) Σ_τ ρ_τ(y, q_τ). For a point forecast it reduces to the absolute error.
- **MAE** of the median q_0.50, as the point-accuracy score.
- **Pinball loss by τ**, to see which part of the distribution fails.
- **Interval coverage** of the central 50, 80, 90 and 98 % intervals
  [q_0.25, q_0.75], [q_0.10, q_0.90], [q_0.05, q_0.95], [q_0.01, q_0.99]: the fraction
  of MTUs whose price lies inside. Mean interval widths are reported alongside.
- **PIT histogram** (probability integral transform): PIT = fraction of the 99 quantiles
  below y, randomised uniformly within its 1 % bin; 20 bins. Flat = calibrated.

Aggregation: average over the MTUs of a day, then over days. Every day weighs the same,
whatever its MTU count.

Breakdowns, for every score: regime (R0–R3); local hour of day; and two price subsets,
**negative prices** (y < 0) and **price spikes** (y above the 99th percentile of the
prices of the 365 days before D, a threshold known at issue time). Spikes and negative
prices are never removed.

## 6. Error bars and tests

- **Bootstrap:** moving-block bootstrap over delivery days, block length 14 days, 2 000
  resamples, 95 % percentile intervals, for every score, coverage and profit. Block
  lengths 7 and 28 days are reported as a sensitivity check.
- **Diebold–Mariano test** on daily loss differentials d_D = CRPS_A(D) − CRPS_B(D)
  (the multivariate form of Ziel & Weron 2018), Newey–West variance with 7 lags,
  one-sided (H1: A has lower expected CRPS). All model pairs on the full test period and
  on each regime. "A beats B" is claimed only if p < 0.05; when picking a best model out
  of several, p-values are Holm-corrected for the number of comparisons.
- **Neural networks:** 5 seeds per output distribution; the spread over seeds is
  reported with every score.

## 7. Battery simulation (step 11)

- Battery: 1 MW power, 2 MWh energy, round-trip efficiency 90 % (√0.9 per direction),
  empty at the start and end of each delivery day, at most one full cycle per day.
- Schedule: linear programme maximising the forecast profit for day D, using the
  forecast mean price of each MTU (mean of the 99 quantiles); settled at the real prices.
- Reported: profit in EUR per MW per year for each model, the perfect-foresight profit
  (upper bound) and the profit difference to M0, with block-bootstrap intervals;
  overall and by regime.

## 8. References

- Lago, Marcjasz, De Schutter, Weron (2021), Applied Energy 293, 116983 (LEAR, epftoolbox).
- Marcjasz, Narajewski, Weron, Ziel (2023), Energy Economics 125, 106843 (distributional NN).
- Nowotarski, Weron (2015), Computational Statistics 30, 791 (quantile regression averaging).
- Uniejewski, Weron, Ziel (2018), IEEE Trans. Power Systems 33, 2219 (variance-stabilising
  transforms).
- Ziel, Weron (2018), Energy Economics 70, 396 (multivariate Diebold–Mariano test).

## Change record

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-11-23 | 4, M0 | "same MTU one week earlier" → "same local hour one week earlier", quarter-hours via the common layer | consistency with section 3; changed before any model was run |
| 2026-11-23 | 4, shared inputs | wind and solar forecasts "for D and D−1" → "for D−1 and D−2" | the publication probe found the D forecasts unpublished at the issue time (D3.4), so the original wording broke the no-peeking rule of section 1; the lags are derived from the publication rules by `features.first_allowed_lag` and were in the code before the first M1–M3 forecast was made; the plan text was corrected afterwards, with this entry |
| 2026-11-23 | 1 | reference "D3.3" → "D3.4" | numbering typo |
| 2026-11-23 | 4, M2 | search space also includes the feature fraction {0.5, 0.8, 1.0} and L2 penalty {0, 1, 10} | two standard LightGBM regularisers; added before any tuning result existed; still 30 configurations |
| 2026-11-24 | 1, 5 | all quantiles of every model are clipped to the SDAC price limits (−500 €/MWh; maximum 3 000, 4 000 from 10 May 2022) before scoring, in the battery and in the daily run | the M3 test forecasts contained non-finite values (25 quantiles of the JSU ensemble) because the inverse transform sinh grows exponentially in the tails, so they could not be scored; found by a finiteness check of the forecast files **after the test forecasts were made and before any test score was computed**. Clipping uses only limits fixed in advance and never raises the pinball loss. Validation scores with clipping: unchanged except M3-JSU 4.325 → 4.324, so no validation choice changes (D10.1) |
