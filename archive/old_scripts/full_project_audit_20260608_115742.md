# Full Project Audit

Generated: 2026-06-08 11:57:42.466605



## Critical files

- `data/processed/model_dataset.csv`: 5,966,230 bytes, modified 2026-06-08 11:52:59.851267

- `data/processed/world_cup_2026_group_stage_features.csv`: 18,462 bytes, modified 2026-06-08 11:14:54.427344

- `data/raw/elo_2026_results.csv`: 17,018 bytes, modified 2026-06-08 11:48:16.914277

- `models/production_model_v3.joblib`: 7,891,086 bytes, modified 2026-06-08 11:14:54.224344

- `models/production_config_v3.json`: 924 bytes, modified 2026-06-08 11:14:53.417344

- `scripts/update_world_cup_data.py`: 4,675 bytes, modified 2026-06-08 11:48:07.113277

- `src/tournament/match_simulation.py`: 4,831 bytes, modified 2026-06-08 11:14:54.432344

- `src/tournament/simulate_world_cup.py`: 10,453 bytes, modified 2026-06-08 11:14:54.433344

- `src/app/streamlit_app.py`: 9,154 bytes, modified 2026-06-08 11:14:54.431344


## Model configs

### models/production_config.json
- model_type: ensemble
- features: 21
- target_cols: ['goals_A', 'goals_B']
- use_weights: True
- trained_on_date: 2026-06-06

### models/production_config_v2.json
- model_type: ensemble
- features: 21
- target_cols: ['goals_A', 'goals_B']
- use_weights: False
- trained_on_date: 2026-06-07

### models/production_config_v3.json
- model_type: ensemble
- features: 20
- target_cols: ['goals_A', 'goals_B']
- use_weights: True
- trained_on_date: 2026-06-07



## CSV summaries

### data/processed/model_dataset.csv
- shape: (21540, 33)
- columns: ['date', 'team_a', 'team_b', 'competition', 'location', 'season_id', 'tournament_year', 'tournament_key', 'rank_diff', 'elo_diff', 'rating_a_before', 'rating_b_before', 'avg_player_value_diff', 'log_market_value_a', 'market_value_rel_mean_diff', 'opponent_strength_diff_last5', 'weighted_goals_for_diff_last5', 'weighted_goals_against_diff_last5', 'rating_change_diff_last5', 'team_a_matches_played_before', 'team_b_matches_played_before', 'team_a_days_since_last_match', 'team_b_days_since_last_match', 'team_a_tournament_matches_played', 'team_b_tournament_matches_played', 'tournament_points_diff', 'tournament_goal_diff_diff', 'goalkeeper_share_diff', 'defender_share_diff', 'target_goals_a', 'target_goals_b', 'target_goal_diff', 'target_total_goals']

head:
```
         date        team_a                team_b                 competition location  season_id  tournament_year                   tournament_key  rank_diff  elo_diff  rating_a_before  rating_b_before  avg_player_value_diff  log_market_value_a  market_value_rel_mean_diff  opponent_strength_diff_last5  weighted_goals_for_diff_last5  weighted_goals_against_diff_last5  rating_change_diff_last5  team_a_matches_played_before  team_b_matches_played_before  team_a_days_since_last_match  team_b_days_since_last_match  team_a_tournament_matches_played  team_b_tournament_matches_played  tournament_points_diff  tournament_goal_diff_diff  goalkeeper_share_diff  defender_share_diff  target_goals_a  target_goals_b  target_goal_diff  target_total_goals
0  2004-01-01       Bermuda              Barbados  Dudley Eve Memorial Trophy  Bermuda       2004             2004  Dudley Eve Memorial Trophy_2004         52       -99             1270             1369                    1.0            0.693147                    0.016384                      0.000000                            0.0                           0.000000                       0.0                             0                             0                            60                            60                                 0                                 0                       0                          0                    0.0                  0.0               0               4                -4                   4
1  2004-01-01        Kuwait                 Yemen                    Gulf Cup   Kuwait       2004             2004                    Gulf Cup_2004        -91       340             1519             1179                    0.0            0.000000                    0.000000                      0.000000                            0.0                           0.000000                       0.0                             0                             0                            60                            60                                 0                                 0                       0                          0                    0.0                  0.0               4               0                 4                   4
2  2004-01-01  Saudi Arabia               Bahrain                    Gulf Cup   Kuwait       2004             2004                    Gulf Cup_2004        -43       139             1662             1523                    0.0            0.000000                    0.000000                      0.000000                            0.0                           0.000000                       0.0                             0                             0                            60                            60                                 0                                 0                       0                          0                    0.0                  0.0               1               0                 1                   1
3  2004-01-03       Bahrain                  Oman                    Gulf Cup   Kuwait       2004             2004                    Gulf Cup_2004        -10       -49             1511             1560                    0.0            0.000000                    0.000000                    181.311955                            0.0                           0.890907                     -12.0                             1                             0                             2                            60                                 1                                 0                       0                         -1                    0.0                  0.0               1               0                 1                   1
4  2004-01-03         Qatar  United Arab Emirates                    Gulf Cup   Kuwait       2004             2004                    Gulf Cup_2004         -8        66             1521             1455                    0.0            0.000000                    0.000000                      0.000000                            0.0                           0.000000                       0.0                             0                             0                            60                            60                                 0                                 0                       0                          0                    0.0                  0.0               0               0                 0                   0
```

missing values top:
```
date                                 0
team_a                               0
team_b                               0
competition                          0
location                             0
season_id                            0
tournament_year                      0
tournament_key                       0
rank_diff                            0
elo_diff                             0
rating_a_before                      0
rating_b_before                      0
avg_player_value_diff                0
log_market_value_a                   0
market_value_rel_mean_diff           0
opponent_strength_diff_last5         0
weighted_goals_for_diff_last5        0
weighted_goals_against_diff_last5    0
rating_change_diff_last5             0
team_a_matches_played_before         0
```
- unique teams: 244

### data/processed/world_cup_2026_group_stage_features.csv
- shape: (72, 26)
- columns: ['match_id', 'date', 'team_a', 'team_b', 'group', 'rank_diff', 'elo_diff', 'rating_a_before', 'rating_b_before', 'avg_player_value_diff', 'opponent_strength_diff_last5', 'weighted_goals_for_diff_last5', 'log_market_value_a', 'weighted_goals_against_diff_last5', 'team_b_matches_played_before', 'team_a_matches_played_before', 'market_value_rel_mean_diff', 'rating_change_diff_last5', 'defender_share_diff', 'goalkeeper_share_diff', 'team_b_days_since_last_match', 'team_a_days_since_last_match', 'tournament_goal_diff_diff', 'tournament_points_diff', 'team_a_tournament_matches_played', 'team_b_tournament_matches_played']

head:
```
   match_id                       date         team_a                  team_b    group  rank_diff  elo_diff  rating_a_before  rating_b_before  avg_player_value_diff  opponent_strength_diff_last5  weighted_goals_for_diff_last5  log_market_value_a  weighted_goals_against_diff_last5  team_b_matches_played_before  team_a_matches_played_before  market_value_rel_mean_diff  rating_change_diff_last5  defender_share_diff  goalkeeper_share_diff  team_b_days_since_last_match  team_a_days_since_last_match  tournament_goal_diff_diff  tournament_points_diff  team_a_tournament_matches_played  team_b_tournament_matches_played
0         1  2026-06-11 19:00:00+00:00         Mexico            South Africa  Group A        -59     334.0           1858.0           1524.0                   4.98                         159.4                       0.326474            4.461300                          -1.347439                           322                           404                    0.299216                      11.4            -0.171540              -0.008528                            60                            60                          0                       0                                 0                                 0
1         2  2026-06-12 02:00:00+00:00    South Korea                 Czechia  Group A         -8      26.0           1752.0           1726.0                  -1.25                         351.8                      -0.920258            4.972587                          -0.224675                           247                           327                   -0.482293                       6.2            -0.066928              -0.131421                            60                            60                          0                       0                                 0                                 0
2         3  2026-06-12 19:00:00+00:00         Canada  Bosnia and Herzegovina  Group B        -41     190.0           1784.0           1594.0                   6.95                           5.0                      -0.995089            5.409188                          -0.525843                           209                           224                    2.023121                     -17.6             0.334893               0.029796                            60                            60                          0                       0                                 0                                 0
3         4  2026-06-13 01:00:00+00:00  United States                Paraguay  Group D         19    -112.0           1721.0           1833.0                   7.68                          97.4                       1.562670            5.879695                           0.410570                           254                           368                    1.986287                       6.6             0.030985              -0.028036                            60                            60                          0                       0                                 0                                 0
4         8  2026-06-13 19:00:00+00:00          Qatar             Switzerland  Group B         80    -464.0           1425.0           1889.0                 -11.49                        -298.2                      -1.104634            3.066657                           0.616860                           248                           358                   -2.702315                     -10.2            -0.040947              -0.058027                            60                            60                          0                       0                                 0                                 0
```

missing values top:
```
match_id                             0
date                                 0
team_a                               0
team_b                               0
group                                0
rank_diff                            0
elo_diff                             0
rating_a_before                      0
rating_b_before                      0
avg_player_value_diff                0
opponent_strength_diff_last5         0
weighted_goals_for_diff_last5        0
log_market_value_a                   0
weighted_goals_against_diff_last5    0
team_b_matches_played_before         0
team_a_matches_played_before         0
market_value_rel_mean_diff           0
rating_change_diff_last5             0
defender_share_diff                  0
goalkeeper_share_diff                0
```
- unique teams: 48

### outputs/evaluation/world_cup_2026_simulation/group_predictions.csv
- shape: (72, 17)
- columns: ['match_id', 'date', 'group', 'team_a', 'team_b', 'lambda_a', 'lambda_b', 'pred_goals_a', 'pred_goals_b', 'pred_score', 'team_a_win_prob', 'draw_prob', 'team_b_win_prob', 'team_a_tournament_matches_played', 'team_b_tournament_matches_played', 'tournament_points_diff', 'tournament_goal_diff_diff']

head:
```
   match_id                       date    group         team_a                  team_b  lambda_a  lambda_b  pred_goals_a  pred_goals_b pred_score  team_a_win_prob  draw_prob  team_b_win_prob  team_a_tournament_matches_played  team_b_tournament_matches_played  tournament_points_diff  tournament_goal_diff_diff
0         1  2026-06-11 19:00:00+00:00  Group A         Mexico            South Africa  2.060667  0.504301             2             0        2-0         0.740720   0.180186         0.078799                                 0                                 0                       0                          0
1         2  2026-06-12 02:00:00+00:00  Group A    South Korea                 Czechia  1.331356  0.863877             1             0        1-0         0.475937   0.281256         0.242795                                 0                                 0                       0                          0
2         3  2026-06-12 19:00:00+00:00  Group B         Canada  Bosnia and Herzegovina  2.060153  0.497960             2             0        2-0         0.742255   0.179761         0.077690                                 0                                 0                       0                          0
3         4  2026-06-13 01:00:00+00:00  Group D  United States                Paraguay  0.804516  1.179585             0             1        0-1         0.250922   0.302419         0.446655                                 0                                 0                       0                          0
4         8  2026-06-13 19:00:00+00:00  Group B          Qatar             Switzerland  0.460943  2.607513             0             2        0-2         0.045542   0.121583         0.831359                                 0                                 0                       0                          0
```

missing values top:
```
match_id                            0
date                                0
group                               0
team_a                              0
team_b                              0
lambda_a                            0
lambda_b                            0
pred_goals_a                        0
pred_goals_b                        0
pred_score                          0
team_a_win_prob                     0
draw_prob                           0
team_b_win_prob                     0
team_a_tournament_matches_played    0
team_b_tournament_matches_played    0
tournament_points_diff              0
tournament_goal_diff_diff           0
```
- unique teams: 48

score distribution:
```
pred_score
1-0    18
0-1    18
2-0    14
0-2    10
3-0     5
1-1     4
0-0     3
```

### outputs/evaluation/world_cup_2026_simulation/group_standings.csv
- shape: (48, 11)
- columns: ['group', 'team', 'played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against', 'goal_diff', 'points', 'position']

head:
```
     group          team  played  wins  draws  losses  goals_for  goals_against  goal_diff  points  position
0  Group A   South Korea       3     3      0       0          3              0          3       9         1
1  Group A        Mexico       3     2      0       1          3              1          2       6         2
2  Group A       Czechia       3     1      0       2          1              2         -1       3         3
3  Group A  South Africa       3     0      0       3          0              4         -4       0         4
4  Group B        Canada       3     2      1       0          5              1          4       7         1
```

missing values top:
```
group            0
team             0
played           0
wins             0
draws            0
losses           0
goals_for        0
goals_against    0
goal_diff        0
points           0
position         0
```

### outputs/evaluation/world_cup_2026_simulation/knockout_results.csv
- shape: (32, 14)
- columns: ['team_a', 'team_b', 'lambda_a', 'lambda_b', 'goals_a', 'goals_b', 'pred_score', 'team_a_win_prob', 'draw_prob', 'team_b_win_prob', 'winner', 'loser', 'round', 'match_slot']

head:
```
        team_a       team_b  lambda_a  lambda_b  goals_a  goals_b pred_score  team_a_win_prob  draw_prob  team_b_win_prob       winner     loser round match_slot
0      Ecuador     Paraguay  1.472906  0.671797        1        0        1-0         0.565206   0.264993         0.169777      Ecuador  Paraguay   R32     R32_01
1    Argentina      Morocco  2.176730  0.691642        2        0        2-0         0.714742   0.180394         0.104429    Argentina   Morocco   R32     R32_02
2       Mexico  Switzerland  0.943073  0.972783        0        0        0-0         0.333785   0.316593         0.349621  Switzerland    Mexico   R32     R32_03
3  Netherlands     Scotland  1.449208  0.753814        1        0        1-0         0.536817   0.268393         0.194768  Netherlands  Scotland   R32     R32_04
4     Portugal      Croatia  1.521457  1.138980        1        1        1-1         0.460712   0.254850         0.284403     Portugal   Croatia   R32     R32_05
```

missing values top:
```
team_a             0
team_b             0
lambda_a           0
lambda_b           0
goals_a            0
goals_b            0
pred_score         0
team_a_win_prob    0
draw_prob          0
team_b_win_prob    0
winner             0
loser              0
round              0
match_slot         0
```
- unique teams: 32

score distribution:
```
pred_score
1-0    11
1-1    10
0-1     6
0-0     3
2-0     2
```


## Command checks

```text

$ git status

RETURN CODE: 0

STDOUT:
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   data/processed/model_dataset.csv
	modified:   data/raw/elo_2026_results.csv
	modified:   outputs/evaluation/feature_importance.csv
	modified:   outputs/evaluation/world_cup_backtest_results.csv
	modified:   outputs/evaluation/world_cup_predictions.csv

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	data/backups/
	data/raw/world_cup_updates/
	notebooks/05_build_model_dataset_executed.ipynb
	scripts/audit/full_project_audit.py
	scripts/update_world_cup_data.py

no changes added to commit (use "git add" and/or "git commit -a")


STDERR:



```

```text

$ git log --oneline -8

RETURN CODE: 0

STDOUT:
8374aa6 Sync all knockout simulation changes
9428099 Rebuild stable 2026 knockout simulation pipeline
c965b14 Merge remote-tracking branch 'origin/main' into roee-knockout-simulator
c76086a model version 3 - one model for both goals
879985e Use production model v2 in app and simulations
8539efe Merge remote-tracking branch 'origin/main' into roee-knockout-simulator
df117ef Finalize 2026 simulation data and scripts
26fdf93 Update group simulation with live tournament features


STDERR:



```

```text

$ /home/codespace/.python/current/bin/python -m py_compile src/tournament/match_simulation.py src/tournament/simulate_world_cup.py scripts/simulate_2026_world_cup.py scripts/update_world_cup_data.py

RETURN CODE: 0

STDOUT:


STDERR:



```

```text

$ /home/codespace/.python/current/bin/python scripts/simulate_2026_world_cup.py

RETURN CODE: 0

STDOUT:
================================================================================
WORLD CUP 2026 SIMULATION
================================================================================
Champion: Brazil
Runner-up: Portugal
Third place: Colombia

Final:
  team_a team_b pred_score winner
Portugal Brazil        1-1 Brazil

Saved outputs to: outputs/evaluation/world_cup_2026_simulation


STDERR:



```

```text

$ /home/codespace/.python/current/bin/python scripts/audit/audit_2026_simulation.py

RETURN CODE: 0

STDOUT:
================================================================================
FEATURE CHECK
================================================================================
config == FEATURE_COLS: True
missing in fixtures: []
missing in model_df: []

================================================================================
GROUP PREDICTION CHECK
================================================================================
rows: 72
avg lambda a/b: 1.456 1.094
avg goals a/b: 0.903 0.583

score distribution:
pred_score
1-0    18
0-1    18
2-0    14
0-2    10
3-0     5
1-1     4
0-0     3
Name: count, dtype: int64

Group standings:
  group  position                   team  points  goal_diff  goals_for
Group A         1            South Korea       9          3          3
Group A         2                 Mexico       6          2          3
Group A         3                Czechia       3         -1          1
Group A         4           South Africa       0         -4          0
Group B         1                 Canada       7          4          5
Group B         2            Switzerland       7          3          4
Group B         3                  Qatar       3         -3          1
Group B         4 Bosnia and Herzegovina       0         -4          0
Group C         1                 Brazil       9          7          7
Group C         2               Scotland       6          0          2
Group C         3                Morocco       3         -2          1
Group C         4                  Haiti       0         -5          0
Group D         1                 Turkey       9          3          3
Group D         2              Australia       4          0          1
Group D         3               Paraguay       4          0          1
Group D         4          United States       0         -3          0
Group E         1                Ecuador       7          4          5
Group E         2                Germany       7          4          5
Group E         3            Ivory Coast       3         -3          1
Group E         4                Curaçao       0         -5          0
Group F         1            Netherlands       9          5          5
Group F         2                  Japan       6          2          3
Group F         3                 Sweden       3         -2          1
Group F         4                Tunisia       0         -5          0
Group G         1                Belgium       9          3          3
Group G         2                   Iran       6          1          2
Group G         3                  Egypt       1         -2          0
Group G         4            New Zealand       1         -2          0
Group H         1                  Spain       9          7          7
Group H         2                Uruguay       6          2          3
Group H         3             Cape Verde       1         -4          0
Group H         4           Saudi Arabia       1         -5          0
Group I         1                 France       9          5          5
Group I         2                 Norway       6          2          3
Group I         3                Senegal       3         -1          1
Group I         4                   Iraq       0         -6          0
Group J         1              Argentina       9          6          6
Group J         2                Austria       6          0          2
Group J         3                Algeria       3         -2          1
Group J         4                 Jordan       0         -4          0
Group K         1               Colombia       7          4          5
Group K         2               Portugal       7          4          5
Group K         3             Uzbekistan       3         -3          1
Group K         4               DR Congo       0         -5          0
Group L         1                England       7          5          6
Group L         2                Croatia       7          3          4
Group L         3                 Panama       3         -2          1
Group L         4                  Ghana       0         -6          0

================================================================================
KNOCKOUT CHECK
================================================================================
      round  match_slot      team_a      team_b  lambda_a  lambda_b pred_score  team_a_win_prob  draw_prob  team_b_win_prob      winner
        R32      R32_01     Ecuador    Paraguay  1.472906  0.671797        1-0         0.565206   0.264993         0.169777     Ecuador
        R32      R32_02   Argentina     Morocco  2.176730  0.691642        2-0         0.714742   0.180394         0.104429   Argentina
        R32      R32_03      Mexico Switzerland  0.943073  0.972783        0-0         0.333785   0.316593         0.349621 Switzerland
        R32      R32_04 Netherlands    Scotland  1.449208  0.753814        1-0         0.536817   0.268393         0.194768 Netherlands
        R32      R32_05    Portugal     Croatia  1.521457  1.138980        1-1         0.460712   0.254850         0.284403    Portugal
        R32      R32_06       Spain     Austria  2.208322  0.638493        2-0         0.733195   0.173631         0.092692       Spain
        R32      R32_07      Turkey     Senegal  1.166824  0.944772        1-0         0.408680   0.296086         0.295230      Turkey
        R32      R32_08     Belgium     Czechia  0.843687  1.153548        0-1         0.267377   0.303618         0.429001     Czechia
        R32      R32_09      Brazil       Japan  1.597711  0.793244        1-0         0.564885   0.249928         0.185142      Brazil
        R32      R32_10     Germany      Norway  1.293708  1.099107        1-1         0.409440   0.275466         0.315083     Germany
        R32      R32_11 South Korea      Sweden  1.080486  0.991185        1-0         0.372233   0.301610         0.326153 South Korea
        R32      R32_12     England     Algeria  1.580252  0.786029        1-0         0.562391   0.251971         0.185597     England
        R32      R32_13      France     Uruguay  1.625851  0.729089        1-0         0.588739   0.245461         0.165749      France
        R32      R32_14   Australia        Iran  0.799451  0.932200        0-0         0.295392   0.335603         0.369004        Iran
        R32      R32_15      Canada Ivory Coast  0.763223  0.920751        0-0         0.285422   0.340876         0.373700 Ivory Coast
        R32      R32_16    Colombia      Panama  1.279061  1.154549        1-1         0.393143   0.273698         0.333148    Colombia
        R16      R16_01     Ecuador   Argentina  0.877137  1.191134        0-1         0.270663   0.297242         0.432090   Argentina
        R16      R16_02 Switzerland Netherlands  0.836122  1.620380        0-1         0.192841   0.247686         0.559424 Netherlands
        R16      R16_03    Portugal       Spain  1.294180  1.049573        1-1         0.420831   0.277869         0.301290    Portugal
        R16      R16_04      Turkey     Czechia  0.958222  1.081132        0-1         0.316039   0.304105         0.379853     Czechia
        R16      R16_05      Brazil     Germany  1.656313  0.952229        1-0         0.539010   0.244454         0.216476      Brazil
        R16      R16_06 South Korea     England  0.778395  1.455531        0-1         0.200475   0.267476         0.532027     England
        R16      R16_07      France        Iran  1.571565  0.671906        1-0         0.590630   0.251396         0.157935      France
        R16      R16_08 Ivory Coast    Colombia  1.012858  1.061969        1-1         0.336522   0.301621         0.361854    Colombia
         QF       QF_01   Argentina Netherlands  1.169056  1.156780        1-1         0.362110   0.281809         0.356074   Argentina
         QF       QF_02    Portugal     Czechia  1.102726  0.914393        1-0         0.396544   0.305037         0.298417    Portugal
         QF       QF_03      Brazil     England  1.580155  1.250079        1-1         0.450421   0.247414         0.302117      Brazil
         QF       QF_04      France    Colombia  1.470157  1.743629        1-1         0.326098   0.231467         0.442324    Colombia
         SF       SF_01   Argentina    Portugal  1.003130  1.616485        1-1         0.234850   0.248364         0.516736    Portugal
         SF       SF_02      Brazil    Colombia  1.334020  0.867646        1-0         0.475739   0.280830         0.243420      Brazil
      FINAL       FINAL    Portugal      Brazil  1.102286  1.268840        1-1         0.320745   0.277390         0.401855      Brazil
THIRD_PLACE THIRD_PLACE   Argentina    Colombia  0.967416  1.256948        0-1         0.285026   0.285434         0.429532    Colombia

Knockout score distribution:
pred_score
1-0    11
1-1    10
0-1     6
0-0     3
2-0     2
Name: count, dtype: int64

Champion: Brazil
Runner-up: Portugal
Third: Colombia

================================================================================
SUSPICIOUS MATCHES
================================================================================
No winner/probability contradictions found.


STDERR:



```