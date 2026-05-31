# src/state/ — Tournament State Tracking for Live Use

## Status: Partially implemented.
- `update_state.py`: FUNCTIONAL
- `team_state.py`: STUB
- `group_table.py`: STUB

---

## What's Implemented

### update_state.py
`update_state_after_match(team_states: dict, match_result: dict) → dict`
- Updates running tournament state for both teams after a match result
- State dict per team: played, wins, draws, losses, goals_for, goals_against, points, goal_diff
- Scoring: win=3 pts, draw=1 pt, loss=0 pts
- Input match_result: {team_a, team_b, goals_a, goals_b}
- Returns updated team_states dict

---

## What Needs to Be Implemented

### team_state.py
```
initialize_team_states(teams: list) → dict[str, dict]
  Creates initial zero-state for all tournament teams:
  {team: {played:0, wins:0, draws:0, losses:0, goals_for:0, goals_against:0, points:0, goal_diff:0}}

get_team_state(team_states: dict, team: str) → dict
  Returns state for a team; returns default zero-state if team not found

save_state(team_states: dict, path: str) → None
load_state(path: str) → dict
  JSON persistence between sessions (needed if prediction runs are not continuous)
```

### group_table.py
```
build_group_table(team_states: dict, group_assignments: dict) → pd.DataFrame
  group_assignments: {team_name: group_letter}
  Returns sorted group standings: team, group, P, W, D, L, GF, GA, GD, Pts

get_group_standings(group_table: pd.DataFrame, group: str) → pd.DataFrame
  Returns sorted standings for one group

compute_tournament_points_diff(team_a: str, team_b: str, team_states: dict) → float
  Returns team_a_points - team_b_points
  Used directly as tournament_points_diff feature

compute_tournament_goal_diff_diff(team_a: str, team_b: str, team_states: dict) → float
  Returns team_a_goal_diff - team_b_goal_diff
  Used directly as tournament_goal_diff_diff feature
```

---

## State Flow During Live Tournament
```
Tournament start:
  team_states = initialize_team_states(all_wc_teams)

Before each match:
  feats = build_pre_match_features(..., team_states=team_states)
  prediction = model.predict(feats)

After each match result:
  team_states = update_state_after_match(team_states, {team_a, team_b, goals_a, goals_b})

(save_state periodically for crash recovery)
```

---

## Key Constraint
Tournament state features are 0 for each team's first match — this is the correct behavior and matches how the model was trained. Do not pre-populate from historical tournament data.
For knockout rounds, these features reflect only the current tournament's group stage history.
