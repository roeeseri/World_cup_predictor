# src/prediction/ — Prediction Pipeline

## Status: Implemented and functional.

---

## score_conversion.py
Converts model's continuous expected goals output (λ_a, λ_b) into a discrete predicted score.

### Functions

`poisson_score_grid(lambda_a, lambda_b, max_goals=6) → pd.DataFrame (7×7)`
- P(score_a=i, score_b=j) = Poisson(i | λ_a) × Poisson(j | λ_b)
- Assumes goals by each team are independent (standard assumption)

`most_likely_score(lambda_a, lambda_b) → (int, int)`
- Returns (score_a, score_b) with maximum joint probability from the 7×7 grid
- Handles NaN/inf inputs: clamps to 0 before computing

`outcome_probabilities(lambda_a, lambda_b) → dict`
- Returns {"home_win": float, "draw": float, "away_win": float}
- Derived from summing relevant cells of the Poisson grid

`convert_expected_goals_to_scores(y_pred, method="poisson", max_goals=6) → ndarray (n, 2)`
- Batch conversion for n matches
- method="poisson": uses most_likely_score (preferred)
- method="round": simple rounding (faster, less accurate)
- Applies nan_to_num before processing — safe against NaN model outputs

### Why Poisson Grid Over Simple Rounding
Rounding λ_a=1.3 → 1 ignores that the joint distribution P(1,1) may exceed P(1,0).
The Poisson grid considers all (i, j) pairs and finds the globally most probable score.
This is more principled and empirically better for exact score prediction.

---

## predict_match.py
`predict_match(match: dict, model, team_state: dict) → dict`
- Input: match metadata (team_a, team_b, date, competition, ...) + fitted model + current team_state
- Output: {lambda_a, lambda_b, predicted_score_a, predicted_score_b, home_win_prob, draw_prob, away_win_prob}
- Internally: builds features → model.predict() → score_conversion

---

## Live Tournament Usage Pattern
```python
# Before each match:
features   = build_pre_match_features(match, team_states, ...)  # src/features/
y_pred     = model.predict(features)                            # → ndarray (1, 2)
score      = convert_expected_goals_to_scores(y_pred)[0]        # → (int, int)
probs      = outcome_probabilities(y_pred[0,0], y_pred[0,1])    # → dict

# After match result is known:
update_state_after_match(team_states, match_result)             # src/state/
```
