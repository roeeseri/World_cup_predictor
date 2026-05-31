# src/evaluation/ — Evaluation Infrastructure

## Status: Implemented and functional.

---

## metrics.py
All metric functions take `(y_true: ndarray, y_pred: ndarray)` where shape = (n_matches, 2), columns = [goals_a, goals_b].

| Function | Returns | Notes |
|---|---|---|
| `goal_mae_raw` | float | MAE on expected goals (λ, not rounded) |
| `goal_rmse_raw` | float | RMSE on expected goals |
| `rounded_score_mae` | float | MAE after rounding to integer goals |
| `exact_score_accuracy` | float 0–1 | Fraction of matches with exact score correct |
| `result_accuracy` | float 0–1 | Fraction with correct win/draw/loss |
| `goal_difference_mae` | float | MAE on goal difference |
| `winner_aware_error` | float | MSE with 2× penalty for wrong winner |
| `detect_goal_anomalies` | ndarray[bool] | True where \|goal_diff\| > 4 |
| `weight_anomalies` | ndarray[float] | 0.3 weight for anomalous matches, 1.0 otherwise |

Primary metric: `exact_score_accuracy`. Target for WC: 35–40%.

---

## backtest.py
`run_chronological_backtest(matches_df, model, team_states) → pd.DataFrame`
- Iterates matches in date order
- Before each match: generates prediction using current state
- After each match: updates team state
- Returns DataFrame with match metadata + predicted scores + actual scores

---

## reports.py
`summarize_evaluation(predictions_df) → dict`
- Computes all 7 metrics from prediction DataFrame
- Returns `{metric_name: value}`

---

## Usage Pattern
```python
results = run_chronological_backtest(wc_matches, trained_model, initial_states)
report  = summarize_evaluation(results)
# report['exact_score_accuracy'] is the primary number to track
```
