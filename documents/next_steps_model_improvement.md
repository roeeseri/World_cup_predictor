# Next Steps: Reaching 40% Exact Score Accuracy

Current honest baseline: ~30% exact score (LightGBM, competition-weighted, no test-set tuning).
Optuna-tuned ceiling on 2022 WC: 37.5% — but this was tuned on the test year, not a real estimate.
Target: 40% exact score on a properly held-out WC year.

---

## Priority 1 — Fix Evaluation First

**Nothing else matters until this is done. Current numbers are unreliable.**

### Proper WC-year cross-validation
```
Fold 1: train 2004-2009  →  evaluate on 2010 WC
Fold 2: train 2004-2013  →  evaluate on 2014 WC
Fold 3: train 2004-2017  →  evaluate on 2018 WC   ← use this for hyperparameter tuning
Fold 4: train 2004-2021  →  evaluate on 2022 WC   ← hold out until final evaluation
```
Average exact score across all 4 folds = real model accuracy.
Tune hyperparameters using fold 3 (2018 WC) only. Report fold 4 (2022 WC) as the held-out test — touch it once.

### Metrics to add alongside exact score
- **RPS (Ranked Probability Score)** — proper metric for win/draw/loss probability. Penalizes confident wrong predictions more than uncertain ones. Better than result accuracy for evaluating probability outputs.
- **Calibration plot** — predicted P(win_a)=70% should match actual win rate ~70%. Plot predicted probability buckets vs observed frequency.
- **Brier score** — proper scoring rule on the 3-way result. Comparable across models and time.
- **Log-loss on score probabilities** — if outputting full score distributions.

---

## Priority 2 — xG Form Features

Replace `weighted_goals_for/against_diff_last5` with **xG (expected goals)** equivalents.

**Why**: Actual goals are noisy. A team that scored 0 but created 2.8 xG is fundamentally different from a team that created nothing. xG-based form is a much cleaner signal of team quality and current shape.

**Source**: FBref (free, goes back to ~2017 for most competitions), Understat (club competitions). WC qualifier xG is available from FBref.

**Implementation**:
- Scrape match-level xG for team_a and team_b from FBref
- Join to historical match dataset by (team, date)
- Replace actual goal-based form features with xG-based equivalents
- Add `xg_for_diff_last5`, `xg_against_diff_last5` as new FEATURE_COLS

**Expected impact**: Medium-high. xG form features should outperform actual goals form for future match prediction because they correct for luck (penalties, long-shots, goalkeeper saves).

---

## Priority 3 — Tournament as Time Series (In-tournament Bayesian Update)

**The core insight**: the model currently treats each match as independent given pre-tournament features. But a team that just beat Germany 2-1 is genuinely stronger than its pre-tournament ELO suggests — not just because of the 3 points, but because we now have new information about their actual current form.

### How it works
- **Pre-tournament prior**: ELO, market value, form → model predicts lambda_a and lambda_b
- **In-tournament update**: after each match, update a running "tournament attack strength" per team based on actual goals vs expected goals
- **Blend**: `lambda_final = (1 - w) * prior_lambda + w * tournament_lambda`
  where w increases as more tournament matches are played (0 for match 1, 0.2 for match 2, 0.35 for match 3...)

### Update rule (ELO-style for goals)
```
expected_goals = prior_lambda
actual_goals   = match result
update         = K * (actual_goals - expected_goals)
new_attack_strength = old_attack_strength + update
```
K is a learning rate (~0.3-0.5). This is equivalent to an ELO update but on goals rather than results.

### What this captures that current state features miss
- **Momentum**: a team on a scoring run genuinely performs better
- **Tactical evolution**: teams adapt their style as tournament progresses
- **Confidence**: squads playing freely after early wins
- **Fatigue**: teams that went to extra time in the previous round
- **Squad rotation signals**: if a strong team rotated in match 3, their tournament lambda reflects it

### Implementation plan
1. Add `tournament_lambda_a` and `tournament_lambda_b` as new features (initialized to prior prediction, updated after each match)
2. Add `tournament_goals_for_a`, `tournament_goals_against_a` as running totals
3. Blend with prior based on matches played: `blend_weight = min(0.4, matches_played * 0.15)`
4. Train model with these features included — they will have 0 value for match 1 of each team, non-zero from match 2 onwards

---

## Priority 4 — Additional Data Sources

### xG (see Priority 2)
Already covered above. Biggest single data improvement.

### Head-to-head history
**Easy to compute from existing dataset.**
- Win/draw/loss rates between specific team pairs (last 10 H2H matches)
- Average goals scored/conceded in H2H
- Some matchups consistently defy ELO (e.g. underdogs with strong defensive setups vs historically dominant teams)
- Features: `h2h_win_rate_a`, `h2h_avg_goals_a`, `h2h_matches_count`
- Discount by recency and relevance (competitive vs friendly H2H)

### Squad experience
**Source**: Transfermarkt (already scraping), WC official site.
- Average caps per player in squad
- Number of players with prior WC experience
- Average age (peak performance window is 26-29 for outfield players)
- Features: `avg_caps_diff`, `wc_experience_diff`, `squad_age_diff`

### Key player availability
**Source**: WC official match reports, transfermarkt injury history.
- Binary: is the top scorer / key midfielder available for this match?
- Suspension flag (yellow card accumulation — fully predictable pre-match)
- This is the hardest to get historically but most predictive for individual matches

### Bench depth
**Source**: Transfermarkt (already have market values per position).
- Gap between starting XI value and squad total value = bench depth
- Teams with deeper benches maintain performance in later rounds
- Feature: `bench_depth_diff` = (total_squad_value - starting_xi_value) differential

### Pressing / tactical stats
**Source**: FBref (PPDA — passes allowed per defensive action).
- High-press teams create more chances but tire faster in heat/altitude
- Feature: `ppda_diff_last5`

---

## Priority 5 — Hyperparameter Tuning (Properly)

### What to tune
- LightGBM tree params (already done, but retune on fold 3 not fold 4)
- **Competition weights** — the 5x/3x/1x ratios are guesses. Add them to the Optuna search space.
- **Temporal decay rate** — how fast older matches should be down-weighted (currently fixed at 0.95/year)
- **Calibration hold-out ratio** — 0.15 vs 0.25 vs 0.35 of training data
- **Tournament time series blend weight** — how much to trust in-tournament updates

### Tuning target
Use **fold 3 (2018 WC)** as the tuning target. Never tune on fold 4 (2022 WC).
Report fold 4 results once, at the end, as the honest evaluation.

---

## Priority 6 — Dixon-Coles Score Correlation

The independence assumption (goals_a independent of goals_b) is structurally wrong. When one team scores more, the game opens up — affecting the opponent's goal count too.

Dixon-Coles adds a parameter **rho (ρ)** that adjusts the probability of low-scoring tight results:
```
Adjusted P(i, j) = P_poisson(i, j) * tau(i, j, rho)

tau(0, 0, rho) = 1 - lambda_a * lambda_b * rho
tau(1, 0, rho) = 1 + lambda_b * rho
tau(0, 1, rho) = 1 + lambda_a * rho
tau(1, 1, rho) = 1 - rho
tau(i, j, rho) = 1   for i+j > 2
```

Typical rho values: -0.1 to -0.3 (negative = 0-0 and 1-1 are more likely than Poisson predicts).

**Implementation**: add to `src/prediction/score_conversion.py` as `dixon_coles_score_grid(lambda_a, lambda_b, rho)`. Estimate rho from training data by MLE.

**Expected impact**: reduces over-prediction of 2-0 vs 1-1 and 1-0 vs 0-0. Directly addresses the mode collapse problem.

---

## Summary Table

| Priority | Task | Effort | Expected Gain | Risk of Overfit |
|---|---|---|---|---|
| 1 | Proper WC-year CV + metrics | Medium | Gives honest baseline | None — evaluation fix |
| 2 | xG form features | Medium | +2-3% exact score | Low |
| 3 | Tournament time series | High | +2-4% exact score | Medium |
| 4a | H2H features | Low | +0.5-1% | Low |
| 4b | Squad experience features | Low | +0.5-1% | Low |
| 4c | Key player availability | High (data) | +1-2% | Low |
| 5 | Retune HPs on 2018 WC | Low | Honest estimate of tuning gain | Low |
| 6 | Dixon-Coles rho | Medium | +1-2% exact score | Low |

### Realistic accuracy targets
- Current honest baseline: ~30%
- After Priority 1+2+5: ~33% honest
- After Priority 1+2+3+5: ~35-37% honest
- After all priorities: ~38-40% honest

Professional bookmakers with proprietary data reach ~35-40% exact score accuracy on WC matches. The theoretical ceiling given stochasticity of football is approximately 42-45%.

---

## Notes on Current Architecture

- `src/models/lgbm_model.py` — LGBMGoalModel, two separate regressors (goals_a / goals_b)
- `src/features/tournament_state_features.py` — needs enrichment for Priority 3
- `src/prediction/score_conversion.py` — needs Dixon-Coles for Priority 6
- `scripts/demo_2022_wc.py` — demo using current src pipeline, match-by-match simulation
- Current best: 34.4% exact score on 2022 WC (Optuna params, not independently validated)
