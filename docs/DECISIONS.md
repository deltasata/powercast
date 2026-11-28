# Design decisions

One entry per decision point, numbered by step. Format: question; options (pro / con);
choice; evidence; later decisions that depend on it.

## Step 1

### D1.1 Python environment
- **Question:** how to manage the environment inside the repository?
- **Options:**
  - conda: familiar; large, slow, installs outside the directory by default.
  - venv + pip: standard; slow resolver, no lock file.
  - uv-managed venv in `.venv/`: fast, interpreter and cache kept inside the directory.
- **Chosen:** uv-managed `.venv/` with interpreter cached in `.cache/`.
- **Evidence:** rule "environments and installs inside this directory".
- **Affects:** D8.x, D9.x (adding PyTorch and foundation-model packages).

## Step 2

### D2.1 Data sources
- **Question:** which source is primary, which is the cross-check?
- **Options:**
  - SMARD: official (Bundesnetzagentur), CC BY 4.0, 2015 onward; undocumented API.
  - Energy-Charts: documented API, CC BY 4.0; rate-limited; derived from the same TSO data.
  - ENTSO-E: original source; needs registration.
- **Chosen:** SMARD primary, Energy-Charts cross-check, ENTSO-E not used.
- **Evidence:** project rules (ENTSO-E only as backup, after asking the project owner).
- **Affects:** D2.2, D3.2.

### D2.2 Geographic region of the fundamentals
- **Question:** load, wind and solar for Germany only (DE), for the bidding zone (DE-LU),
  or DE + LU?
- **Options:**
  - DE-LU: matches the bidding zone; on SMARD only from 30 Sep 2018.
  - DE + LU summed: matches the zone; LU wind/solar forecasts only from Mar 2022.
  - DE: covers 2015 onward for every series; misses Luxembourg (about 1–2 % of zone load).
- **Chosen:** DE.
- **Evidence:** SMARD index files (LOG step 2, observation 2). One consistent series over
  the whole period matters more than a 1–2 % share of load.
- **Affects:** all models' inputs.

### D2.3 Price history before the zone split (1 Oct 2018)
- **Question:** use the DE-AT-LU price before the split?
- **Options:**
  - Start in Oct 2018: one zone only; 3.75 years less history.
  - Splice DE-AT-LU (filter 251) with DE-LU (filter 4169): longer history, which matters
    for validation years and long training windows; small structural change at the split.
- **Chosen:** splice, keeping a `zone` column.
- **Evidence:** both series are the same auction; the split changes the price level only
  slightly, and the training-window decision (step 4) controls how much old data is used.
- **Affects:** D4.x (validation period, training window).

### D2.4 Fuel prices (gas, CO₂)
- **Question:** include daily gas and CO₂ prices as inputs?
- **Options:**
  - Paid data (EEX, ICE): excluded by the zero-money rule.
  - Scraped websites (Investing.com etc.): their terms forbid reuse.
  - EEX primary-auction EUA prices: free download; reuse terms unclear; not daily.
  - No fuel prices: models rely on recent prices, which carry the fuel cost level.
- **Chosen:** no fuel prices.
- **Evidence:** search of 23 Nov 2026 (LOG step 2); project rules ("continue without them if
  none is free"). The LEAR and distributional-network benchmarks of the literature
  (Lago et al. 2021; Marcjasz et al. 2023) also use no fuel prices.
- **Affects:** model inputs; a stated limitation for the 2021–2022 gas crisis.

## Step 3

### D3.1 Target price series by period
- **Question:** which SMARD file defines the price of each market time unit (MTU)?
- **Options:**
  - Quarter-hour file everywhere: one file; before Oct 2025 it is the hourly price
    repeated four times, i.e. fake 15-minute resolution.
  - Hourly file everywhere: loses the real 15-minute prices after Oct 2025.
  - Hourly file before 1 Oct 2025, quarter-hour file from then on.
- **Chosen:** the third option; each delivery day keeps its true MTU (60 or 15 minutes).
- **Evidence:** LOG step 2, observation 3; LOG step 3 (hourly file = mean of the four
  quarter-hours after the switch, within 0.005 €/MWh).
- **Affects:** D4.x (how models produce 15-minute forecasts).

### D3.2 Units
- **Question:** keep SMARD's MWh per interval or convert to MW?
- **Chosen:** MW (× 4 for quarter-hours), the unit of Energy-Charts and of the
  literature. Evidence: LOG step 2, observation 4.

### D3.3 Missing SMARD prices (20–21 Nov 2026)
- **Question:** how to handle days SMARD has not published?
- **Options:**
  - Leave missing: honest; breaks lagged inputs of the following week.
  - Fill from Energy-Charts: same auction results; agree exactly on 134 163 common prices.
  - Interpolate: invents prices.
- **Chosen:** fill from Energy-Charts, marked in the column `source`.
- **Evidence:** cross-check (LOG step 3): max |difference| 0.00 €/MWh on all common prices.
- **Affects:** daily run (D12.x): the same fallback is used live.

### D3.4 Issue time and publication rules
- **Question:** at what clock time is the forecast issued, and when is each input public?
- **Options for the issue time:**
  - 11:00 on D−1: one hour to submit bids and to absorb the start delay of scheduled
    GitHub Actions jobs (often 10–30 min).
  - 11:55 on D−1: uses everything published before gate closure; not workable for a
    scheduled job, no margin for bidding.
- **Chosen:** 11:00 Berlin on D−1 (as fixed in the evaluation plan before the measurement).
- **Rules:** price of day D public on D−1 at 13:00 (results at about 12:45); load
  forecast for D on D−1 at 10:00 (present on SMARD at 10:10 on 23 Nov; EU Regulation
  543/2013 Art. 6 requires it 2 h before gate closure); wind and solar forecasts for D on
  D−1 at 18:00 (absent on SMARD and Energy-Charts at 11:00 on 23 Nov; on Energy-Charts
  they appeared between 17:59 and 18:04, on SMARD not by 19:57; Art. 14 sets the 18:00
  deadline); measured values 2 h after the end of their interval.
- **Consequence:** the newest wind and solar forecasts at issue time are those for D−1
  and D−2. Many published benchmarks use the target day's forecasts, which in Germany
  are not public on these free sources before the auction; scores here are therefore not
  directly comparable to them.
- **Options not taken (need registration, to ask the owner):** ENTSO-E Transparency
  Platform or netztransparenz.de may publish the TSO forecasts earlier than SMARD.
- **Evidence:** `data/raw/publication_probe/2026-11-24/log.csv` (LOG step 3).
- **Affects:** features of every model (exogenous lags derived from these rules by
  `powercast.features.default_exog_lags`: load D and D−1, wind+solar D−1 and D−2), the
  quarter-hour layer, the daily run.

## Step 4

### D4.1 Validation and test periods
- **Question:** how to split the history into validation and test periods?
- **Options:**
  - Test = last 1–2 years only: standard in the literature; misses the gas crisis.
  - Validation 2019–2020, test 2021-01 – 2026-10: test covers the gas crisis, the
    post-crisis years and the 15-minute period; hyperparameters chosen before all of them.
  - Validation just before the test (e.g. 2023), crisis evaluated as a side backtest:
    tuning closer to the test regime; the crisis backtest would use hyperparameters
    chosen on later data.
- **Chosen:** validation 2019–2020, test 2021-01-01 – 2026-10-31.
- **Evidence:** the project brief asks for results by regime including the gas crisis and the
  15-minute period, all on held-out data; this is the only option that keeps every
  regime strictly after the tuning period.
- **Affects:** all tuning (D5–D9), regime breakdown (D10).

### D4.2 Quarter-hour forecasts after 1 Oct 2025
- **Question:** how do models trained mostly on hourly prices forecast 15-minute prices?
- **Options:**
  - Native 15-minute models: before Oct 2025 the targets are hourly, so intra-hour
    structure is learnt from only 13 months.
  - Hourly-mean forecast + common intra-hour layer driven by the 15-minute
    residual-load forecasts: uses all history for the hourly level; the layer is shared,
    so model comparisons stay fair.
- **Chosen:** hourly mean + common intra-hour layer (EVALUATION_PLAN section 3).
- **Evidence:** LOG step 2, observation 3 (no 15-minute prices before Oct 2025);
  fairness of comparison between models.
- **Affects:** D10 (R3 scores), D12 (daily run).

### D4.3 Score definition
- **Question:** which proper score is primary?
- **Chosen:** CRPS approximated as 2 × mean pinball loss over 99 levels, in €/MWh; it
  equals the absolute error for a point forecast, so it is directly comparable to MAE.
- **Evidence:** the project brief defines CRPS as the average pinball loss; test `tests/test_scores.py` reproduces the closed-form CRPS
  of a normal distribution within 3 %.

## Step 5

### D5.1 Naive error window
- **Question:** over how many past days is the naive rule's error distribution taken?
- **Options:** 28, 91, 182, 364 days (shorter adapts faster, longer estimates tails better).
- **Chosen:** 364 days.
- **Evidence:** lowest validation CRPS (7.978 €/MWh) and coverage closest to nominal (LOG
  step 5); differences to 91 and 182 days are small.

## Step 6

### D6.1 LASSO regularisation strength when samples < features
- **Question:** how to choose the LASSO strength for the 56- and 84-day windows (fewer
  days than the 199 inputs), where current scikit-learn refuses to estimate the noise?
- **Options:**
  - Drop the short windows: departs from the LEAR reference design.
  - Cross-validation: different criterion from the reference; slower.
  - AIC with noise variance = Var(y): what the reference implementation did with the
    scikit-learn of its time.
- **Chosen:** AIC with noise variance = Var(y), for every window.
- **Evidence:** reproduces the epftoolbox behaviour; runs for all windows (LOG step 6).

### D6.2 Quantile-regression solver
- **Question:** which solver fits the 24 × 99 QRA regressions per day?
- **Options:** statsmodels QuantReg (iteratively reweighted least squares); scikit-learn
  QuantileRegressor with the HiGHS linear-programming solver.
- **Chosen:** statsmodels.
- **Evidence:** identical loss to HiGHS on test problems at τ = 0.01, 0.5, 0.95, and
  1.5–30 times faster (23 vs 381 ms at τ = 0.01).

### D6.3 QRA calibration window
- **Question:** how many past days of LEAR point forecasts calibrate the quantile
  regression (plan: chosen on validation from 91, 182 and 364 days)?
- **Options:**
  - 91 days: adapts fastest to a new price level; fewest points for the 1 % and 99 % levels.
  - 182 days: in between.
  - 364 days: most points for the tails; adapts slowest; covers every season once.
- **Chosen:** 364 days.
- **Evidence:** validation 2019–2020 CRPS 4.761 (91), 4.698 (182), **4.643** (364) €/MWh;
  80 % coverage 74.5 / 77.3 / 79.3 %, 98 % coverage 92.1 / 95.8 / 96.8 %. Shorter windows
  give intervals too narrow in the tails.
- **Affects:** test-period M1 run (needs LEAR points from 364 days before 1 Jan 2021),
  live M1 (`config/live.json`).

## Steps 7–9 (compute)

### D7.1 Tuning protocol for LightGBM and the network
- **Question:** how to run a random search of 30 configurations within the compute budget
  (full-cost LightGBM run of the validation period ≈ 105 refits × 82 s)?
- **Options:**
  - Full cost for every configuration: ≈ 70 CPU-hours for LightGBM alone.
  - Proxy: refit every 28 days instead of 7, one seed; LightGBM fits 5 levels
    (0.05, 0.25, 0.5, 0.75, 0.95) and is ranked by their mean pinball loss; the network
    gives all 99 levels at no extra cost and is ranked by CRPS. The winner is then run on
    validation at full cost (refit every 7 days, all levels, 5 seeds for the network).
- **Chosen:** proxy.
- **Evidence:** compute budget; the ranking uses only validation data.
- **Affects:** D10 (final models).

### D7.3 LightGBM configuration
- **Question:** which of the 30 random configurations (D7.1 proxy protocol) is run in full?
- **Options:** the 30 sampled configurations (`results/tuning/gbt_2026-11-23.json`).
- **Chosen:** the lowest mean pinball loss over 5 levels: 15 leaves, learning rate 0.089,
  200 trees, at least 200 samples per leaf, feature fraction 1.0, L2 penalty 10,
  training window 1 092 days (pinball5 = 2.137 €/MWh).
- **Evidence:** validation 2019–2020. Best by window: 1 092 days 2.137, 728 days 2.200,
  364 days 2.318; the top 6 configurations all use 1 092 days and small trees with
  strong regularisation. The best two differ by 0.009 €/MWh, well inside the noise of
  one proxy run, so the choice among them is not meaningful; the rule (lowest score)
  decides.
- **Affects:** validation and test runs of M2.

### D7.2 CPU budget
- **Question:** how many cores may this project use?
- **Chosen:** about 20 parallel worker processes.
- **Evidence:** from 23 Nov ~10:20 the account's nnUNet training jobs of another project
  occupy all six GPUs and many CPU cores; 60 workers pushed the load average to 119 on 64
  cores and would slow those jobs.

### D8.1 Device for the network
- **Question:** GPU or CPU for model M3?
- **Chosen:** CPU (one thread per fit, `CUDA_VISIBLE_DEVICES=""`).
- **Evidence:** all GPUs occupied (D7.2); one fit takes 20–30 s on one CPU thread.

### D8.2 Network configurations
- **Question:** which of the 30 random configurations per output distribution (D7.1
  proxy: refit every 28 days, one seed, ranked by validation CRPS) are run in full?
- **Options:** 30 sampled configurations each (`results/tuning/ddnn_{normal,jsu}_2026-11-23.json`).
- **Chosen:** lowest CRPS per distribution.
  - Normal: hidden layers 512 + 512, ELU, dropout 0.1, learning rate 7.7 × 10⁻⁴,
    L1 6.2 × 10⁻⁴, window 1 092 days (CRPS 4.432 €/MWh).
  - JSU: hidden layers 128 + 512, ELU, dropout 0.3, learning rate 1.0 × 10⁻⁴,
    L1 3.7 × 10⁻⁴, window 1 092 days (CRPS 4.405 €/MWh).
- **Evidence:** validation 2019–2020. The 1 092-day window wins for both (best 728-day:
  4.510 normal, 4.502 JSU). Best and second-best differ by 0.017 (normal) and 0.029
  (JSU) €/MWh, one seed each, so the ranking near the top is within seed noise.
  Three configurations with large learning rates (≥ 3.7 × 10⁻³) diverged (CRPS 14 to
  10²⁰); they lose the ranking and need no special handling.
- **Affects:** validation (5 seeds) and test runs of M3.

### D9.1 Foundation model and its inputs
- **Question:** which pretrained model, and with which inputs?
- **Options:**
  - Chronos-Bolt (univariate only, 9 quantiles 0.1–0.9: no tails for the 98 % interval).
  - TimesFM: univariate by default; covariates through an add-on regression.
  - Chronos-2 (Apache-2.0, 120 M parameters, 8 192-step context, 21 quantiles 0.01–0.99,
    native covariates).
- **Chosen:** Chronos-2, two variants chosen between on validation: prices only, and
  prices + load forecast (known for D) + wind/solar forecast (past only, as its value for
  D is not public).
- **Evidence:** model card and `config.json` of `amazon/chronos-2`; covers the 1 % and
  99 % levels the plan needs.

### D9.2 Chronos-2 variant
- **Question:** prices only, or prices with covariates (D9.1)?
- **Options:**
  - prices only: validation CRPS 4.552 €/MWh, MAE 6.106, 80 % coverage 75.4 %.
  - covariates (load forecast for D as future input; load and wind + solar forecasts as
    past inputs): CRPS 4.419, MAE 5.940, 80 % coverage 77.3 %.
- **Chosen:** covariates.
- **Evidence:** validation 2019–2020, lower CRPS by 0.13 €/MWh, better coverage.
- **Limitation:** Chronos-2 was pretrained by its authors on a large collection of public
  time series. Whether German day-ahead prices of 2019–2026 are in it is not known here,
  so memorisation of the test period cannot be excluded; the zero-shot scores are an
  upper bound on genuine out-of-sample skill in that sense.
- **Affects:** M4 test run, the rule for complex models (plan section 4).

## Step 10

### D10.1 Quantiles outside the auction's price limits
- **Question:** what to do with forecast quantiles outside the possible price range?
  The network (M3) forecasts in the transformed scale z = asinh((p − median)/MAD); the
  price is median + MAD·sinh(z), which grows exponentially, so far-tail quantiles reach
  ±10⁴ to ±∞ €/MWh. Test forecasts: JSU 86–93 days per seed outside [−500, 4 000],
  25 infinite values in the ensemble; normal 7–13 days per seed. Validation: JSU 7–14
  days per seed, normal 0–3.
- **Options:**
  - Score as they are: infinite CRPS on some days, so no mean score exists; the run is
    unusable.
  - Clip in the transformed scale (cap |z|): a new tuning parameter, which would need
    the validation period again.
  - Clip all models' quantiles to the SDAC harmonised clearing-price limits (−500 €/MWh;
    maximum 3 000, 4 000 from delivery 10 May 2022): limits fixed in advance and public,
    no parameter, never raises the pinball loss when the price lies within the limits
    (which it always does).
- **Chosen:** clip to the price limits, for every model, in scoring, battery and the
  daily run (`powercast.evaluate.clip_to_limits`).
- **Evidence:** test `tests/test_limits.py`; validation CRPS with clipping is unchanged
  for M0–M2, M3-normal, M4, and 4.325 → 4.324 for M3-JSU. Limits: SDAC/EPEX SPOT notice
  of 11 Apr 2022 (maximum 4 000 from 10 May 2022), ACER Decision 01/2023 (reference
  limits +4 000 / −500). The highest DE-LU price in the data is 936 €/MWh.
- **Timing:** decided on 24 Nov 2026 after the test forecasts were made, before any
  test score was computed (plan change record).
- **Limitation:** clipping hides, but does not cure, the network's overly heavy tails;
  the share of clipped quantiles is reported with the results.
- **Affects:** all test scores, battery profits, the daily run.

### D10.2 Models of the daily job
- **Question:** which models run every day, and which one is the published forecast?
- **Rule (plan section 4, fixed before training):** M3 or M4 is kept only if it beats
  the best of M0–M2 on the whole test period (lower CRPS and one-sided DM p < 0.05).
- **Test results:** best simple model M1 (CRPS 17.49 €/MWh; M2 19.44). M4 15.42, DM
  p < 0.001 against M1. M3-normal 17.43 (p = 0.40), M3-JSU 17.42 (p = 0.37).
- **Chosen:** M4 (Chronos-2 with covariates) as the published forecast; M1 and M0 run
  alongside as references for the live track record. M3 is not kept.
- **Evidence:** `results/evaluation/test.json`. Check against memorisation: the
  Chronos-2 weights used (commit 29ec3766 of `amazon/chronos-2`) are identical to the
  first published weights of 30 Oct 2025 (same file hash); on 1 Nov 2025 – 31 Oct 2026,
  after publication, M4 still beats M1 (CRPS 16.07 vs 16.97, p = 0.003) and both M3
  (p ≤ 0.05) (`results/evaluation/robustness_test.json`).
- **Affects:** `config/live.json`, the GitHub Actions job (needs PyTorch and the model
  weights, both on CPU).


## Step 12

### D12.1 Making the daily run start in time
- **Question:** GitHub's scheduler started the runs of 24 Nov about 11 h late (19:41 and
  20:20 UTC for the 08:40 and 09:40 UTC slots) and had started none of the 25 Nov slots
  by 10:20 UTC. How to get a forecast before 12:00 Berlin every day?
- **Options:**
  - A. More cron triggers inside 08:15–10:45 UTC: simple; useless if every trigger is
    hours late.
  - B. Local cron job on the cluster as the main runner: on time when the machine is up;
    the machine is shared and not guaranteed to run, and the plan names GitHub Actions.
  - C. External free cron service calling `workflow_dispatch`: on time; needs a
    registration and a token stored at a third party (not allowed without the owner).
  - D. Triggers spread through the night and morning; a run that starts early waits in
    the job until 10:15 Berlin; late or missed days stay in the track record, flagged;
    a local run by hand as fallback.
- **Chosen:** D. Ten triggers at odd minutes (23:17, 02:47, 05:23, 06:53,
  07:37, 08:13, 08:43, 09:13, 09:43, 10:13 UTC); job time limit 350 min.
- **Implementation details added:** the wait is a separate first step, before the
  checkout, so a run that waited checks out the newest `main` and sees a forecast that
  was pushed meanwhile; a run that would wait more than 3.5 h stops at once (3.5 h wait +
  up to 85 min waiting for the load forecast + about 25 min of work fits in 350 min).
  The publish step pulls with rebase before pushing (three attempts). Each forecast
  records `runner` (`github-actions` or `local`), also in `track_record.csv`.
- **Evidence:** run list of the public API (24–25 Nov); GitHub documentation: scheduled
  runs can be delayed under load or dropped, and the start of each hour is busiest.
- **Risks:** on-time delivery is still not guaranteed; the on-time rate must be
  reported from `track_record.csv` (`late`, `runner`). Public repositories stop scheduled
  workflows after 60 days without activity; daily commits prevent this while runs
  succeed.
- **Affects:** `.github/workflows/daily.yml`, `scripts/daily_forecast.py`, README.

### D12.2 Versions of the GitHub actions
- **Question:** the actions used (`checkout@v4`, `setup-python@v5`, `cache@v4`) target
  Node.js 20, which GitHub removed from its runners (changelog of 23 Sep 2026); they now
  run only by being forced onto Node 24. Upgrade or keep?
- **Options:** keep (works today, may break without notice); upgrade to the current
  majors (small changes, none of their breaking changes touch this workflow).
- **Chosen:** upgrade: `checkout@v7` (v7.0.1), `setup-python@v7` (v7.1.0),
  `cache@v6` (v6.1.0), all `node24` (checked on the release pages and `action.yml`).
- **Affects:** `.github/workflows/daily.yml`; first run must confirm that both caches
  are restored and that the push still works.

### D12.3 Hand-over between runs
- **Question:** with D12.1, GitHub started one of ten triggers in the night to 26 Nov
  (06:10 Berlin); it needed a 4 h 05 min wait, more than the 3.5 h cap, and stopped. How
  to turn "some trigger fires at some time of day" into an on-time forecast?
- **Options:**
  - A. Raise the wait cap: the job limit (355 min) minus up to 110 min of work after
    10:15 allows at most about 230 min, which still misses the 26 Nov run.
  - B. Hand-over: a run that is too early sleeps, then starts a fresh run with
    `workflow_dispatch` (GitHub documents that this event, unlike others, does create
    runs when sent with the built-in token), which decides again.
  - B2. A chain that never ends (every run hands over): independent of the scheduler,
    but runners sit idle all day, which GitHub's terms discourage.
  - C. Local runs by hand as the main path.
- **Chosen:** B with a 230 min wait cap, and C as fallback. Each run decides from the
  clock and the forecasts on main (`scripts/gate.py`): target = next delivery day without
  a forecast, issue time 10:15 Berlin the day before; past it: forecast now; within
  230 min: wait and forecast; otherwise sleep min(wait − 180 min, 330 min) and hand over.
  At most 6 hand-overs per chain; a run that forecasts never hands over, so each chain
  ends. Twelve triggers spread over all 24 h, so any one of them seeds the next morning.
- **Evidence:** replay of the runs GitHub actually started (`tests/test_gate.py`): the
  26 Nov 06:10 run hands over once and forecasts at 10:15; the 24 Nov 19:41 run reaches
  the next morning in two hand-overs. Duplicates are prevented by the concurrency group
  (one running, one pending; a new pending run replaces the old one) and by every
  decision depending only on the clock and main.
- **Risks:** a day with no trigger at all still needs the local fallback; the delay of a
  dispatched run is not documented (50 min of slack assumed; to be measured); a failed
  dispatch ends that day's chain until the next trigger.
- **Affects:** `.github/workflows/daily.yml` (`actions: write` permission),
  `scripts/gate.py`, README.

### D12.4 Limiting the hand-over chain to a window before 10:15
- **Question:** with D12.3, a scheduled trigger queued behind a chain starts the next
  chain as soon as the forecast is made, so from 26 Nov 12:31 a runner was busy without
  a break, almost all of it sleeping: in effect option B2, set aside in D12.3 because
  GitHub's terms discourage idle runners. How to keep the on-time forecast and drop the
  idle hours?
- **Options:**
  - A. Keep D12.3: on time on every night seen; a runner busy 24 h a day.
  - B. Window: a run more than W hours before 10:15 stops at once; triggers moved into
    the window. Runner busy at most W h plus about 10 min a day; a night on which GitHub
    starts no trigger inside the window needs the local fallback.
  - C. No hand-over at all (wait cap 230 min only): least idle time, but it would have
    missed the night of 25–26 Nov (one trigger, at 06:10, 4 h 05 min before 10:15).
- **Chosen:** B with W = 8 h (02:15–10:15 Berlin), twice the longest gap to 10:15 seen
  on a night that was served (4 h 05 min). Twelve triggers between 01:17 and 07:53 UTC,
  inside the window in both winter and summer time; a trigger started after 10:15
  forecasts at once, still before the auction if it starts before about 11:30. At most
  2 hand-overs (one suffices inside the window).
- **Evidence:** run list of the public API, 26–28 Nov: GitHub started 22 scheduled runs
  in 44 h (about 11 of 12 a day), mostly 0–3 h after their cron time (the matching of
  runs to cron times is not unique); 4–5 of them fell between 00:00 and 10:15 on each of
  the two nights, one on the night of 25–26 Nov. Replay of all these start times
  (`tests/test_gate.py`): every morning served at 10:15, no chain started outside the
  window.
- **Risks:** only three nights of data; if GitHub starts no trigger between 02:15 and
  about 11:30, that day needs the local fallback. To be checked in the track record
  (`runner`, `late`).
- **Affects:** `scripts/gate.py`, `.github/workflows/daily.yml`, README.
