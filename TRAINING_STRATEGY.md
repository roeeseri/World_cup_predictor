# WC Predictor: Training Strategy & Recommendations

## Executive Summary

**Recommended MVP Approach: Chronological Split with Weighted Training**

For your first running model version, train on all historical data up to a cutoff date, test on future matches, and weight samples by competition importance (World Cup > Regional > Friendlies).

---

## Your Data

- **21,539 matches** from 2004-2026
- **177 different competitions** (mostly friendlies, qualifiers, regional tournaments)
- **43 engineered features** with your partner's feature engineering
- **No actual FIFA World Cup matches yet** (only qualifiers and regional tournaments like Copa America, African Nations Cup)

---

## Three Training Approaches Analyzed

### 1. Train on All Data (Recommended for MVP)
- Train on all matches before a date cutoff
- Test on matches after that date (chronologically forward)
- Apply competition weights (already in dataset)

**Pros:**
- ✅ Simplest to implement
- ✅ Realistic evaluation (how you'll actually predict)
- ✅ No data leakage
- ✅ Leverages full history
- ✅ Iterates fastest

**Cons:**
- ❌ Dilutes World Cup signal with friendlies
- ❌ Older data (2004) may be less relevant

**Use case:** First running version, baseline for comparison

---

### 2. Tournament-Specific Models
- Train ONLY on World Cup qualifiers/tournaments
- Test ONLY on World Cup matches
- Keep regional tournaments separate

**Pros:**
- ✅ Highly specialized for your actual use case
- ✅ Fewer data confounds

**Cons:**
- ❌ Very limited training data (qualifiers only so far)
- ❌ Can't evaluate on full World Cup (no matches yet)
- ❌ May overfit to historical tournament patterns

**Use case:** After MVP works, as a secondary model

---

### 3. Ensemble of Both
- Global model (approach #1) + Tournament model (approach #2)
- Weight predictions by competition type
- Boost ensemble for robustness

**Pros:**
- ✅ Combines best of both worlds
- ✅ Handles uncertainty well

**Cons:**
- ❌ More complex
- ❌ Can't implement until both models exist

**Use case:** Production version after iterating on MVP

---

## Additional Considerations

### Data Weighting Strategy
Your dataset has `competition_weight` column already (e.g., Friendly=1.5, World Cup=?).

**Recommended weights:**
- World Cup matches: 3.0x
- Regional tournaments (Copa America, AFCON): 2.0x
- World Cup qualifiers: 1.5x
- Friendlies: 1.0x

This is already partially in your data via `competition_weight`.

### Time Decay (For Later)
Matches from 2004 are 20+ years old. Consider:
- Down-weight pre-2015 data
- Train two models: "all history" vs "recent only"
- Compare which is better

### Regional Tournament Validation
Before relying on your model for World Cup:
1. Validate on Copa America (test set = next Copa America matches)
2. Validate on African Nations Cup 
3. Validate on European Championship
4. Only then trust World Cup predictions

---

## MVP Implementation (What I Built)

### `notebooks/03_model_experiments.ipynb`

Runs full experimental pipeline:

1. **Feature Analysis** - Ranks 34 features by correlation with goals
2. **Chronological Split** - 80% train / 20% test split by date
3. **Hyperparameter Tuning** - Grid search for best alpha (Poisson) and depth (Tree)
4. **Baseline Models** - Compare against 3 baselines (constant, average, Elo heuristic)
5. **Model Comparison** - Full metrics table with MAE, RMSE, exact score accuracy, result accuracy
6. **Demo Predictions** - Sample predictions + error analysis by competition
7. **Best Model Selection** - Identify which hyperparameters work best

### What You'll See

```
MODEL COMPARISON - Test Set Performance
================================================================================
Model                           Goal MAE  Goal RMSE  Exact Score %  Result Accuracy %
Poisson (α=...)                    ...        ...          ...              ...
Tree (depth=...)                   ...        ...          ...              ...
EloHeuristicBaseline               ...        ...          ...              ...
...
```

Then visualization of:
- Goal prediction error by model
- Match result accuracy (win/draw/loss)
- Exact score prediction rates
- Error distribution and calibration

---

## How to Run the Notebook

```bash
cd notebooks
# Open 03_model_experiments.ipynb in Jupyter or VS Code
# Run all cells sequentially
```

The notebook will:
1. Load your full dataset
2. Infer all numeric features automatically
3. Train models with various hyperparameters
4. Print best hyperparameters for Poisson and Tree
5. Show error analysis per competition
6. Recommend which model to use in production

---

## Next Steps (Recommended Order)

1. **Run the notebook** (today)
   - See where you stand
   - Identify best hyperparameters
   - Check error distribution

2. **Save the best model** (next session)
   - Use the best hyperparameters from notebook
   - Train a production model on all data
   - Save with `src.models.train.save_model()`

3. **Build a demo prediction script** (week 2)
   - Load best model
   - Predict on a few upcoming friendlies
   - Compare predictions to actual results

4. **Add tournament-specific model** (week 3)
   - Duplicate the training logic
   - Filter to World Cup qualifiers only
   - Compare global vs. tournament-specific

5. **Ensemble strategy** (week 4)
   - Combine global + tournament models
   - Validate on regional tournaments

---

## Feature Selection Notes

The notebook identifies top features by correlation with goals. You'll see:

**Likely strong features:**
- `elo_diff` / `rating_a_before` - Team strength difference
- `form_diff_last5` - Recent form
- `market_value_diff` - Squad quality
- `opponent_strength_diff_last5` - Strength of recent opponents
- `tournament_goal_diff_diff` - Current tournament performance
- `competition_weight` - Match importance

**Weak features to consider dropping:**
- Ranked at the bottom of correlation analysis
- May add noise

For next iteration, can run feature importance analysis using tree-based models to refine further.

---

## When Real World Cup Arrives (2026)

Once you have World Cup matches in your test set:

1. **Filter test to World Cup only**
   ```python
   world_cup_matches = df_test[df_test['competition'] == 'FIFA World Cup']
   ```

2. **Measure performance specifically on World Cup**
   - How well does the model predict actual tournament results?
   - Better with tournament-specific or global model?

3. **Backtest strategy**
   - Train on all data before 2026
   - Predict each 2026 group match
   - Compare against actual outcomes
   - Calculate betting value

---

## Key Files Updated

- `notebooks/03_model_experiments.ipynb` - Full experimental pipeline
- This file (`TRAINING_STRATEGY.md`) - Strategy documentation

---

## Questions?

The notebook is self-contained and well-commented. After running it:
- You'll know your best hyperparameters
- You'll see which model family (Poisson vs Tree) works better
- You'll have error analysis by competition
- You'll have demo predictions to validate manually

Good luck! Run the notebook and share results.
