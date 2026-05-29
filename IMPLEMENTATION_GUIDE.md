# Implementation Guide: Training & Evaluation Strategy

This guide explains how to use the new tools for training, tuning, and evaluating your World Cup score prediction models.

---

## What Was Built

### New Modules

1. **`src/models/optuna_tuning.py`** - Optuna-based hyperparameter optimization
   - `PoissonTuner` - Finds best alpha (regularization) for Poisson model
   - `TreeTuner` - Finds best depth and n_estimators for Tree model
   - Optimizes for **exact score accuracy** (your primary metric)

2. **`src/models/feature_analysis.py`** - Feature selection and importance
   - `get_tree_feature_importance()` - Extract feature importance from tree models
   - `correlation_with_exact_score()` - Analyze feature correlations
   - Guides you to best features for production

3. **`src/models/weighting.py`** - Tournament weighting strategies
   - `COMPETITION_WEIGHTS` - Pre-configured weights for major tournaments
   - `apply_competition_weights()` - Weight samples by tournament importance
   - `apply_combined_weighting()` - Combine competition + temporal decay
   - **Major tournaments get 3-5x weight**, friendlies get 1x

4. **`src/models/train_production.py`** - Production training script
   - `train_production_model()` - Train final model with best hyperparameters
   - `evaluate_production_model()` - Evaluate on held-out data
   - Can be run from CLI with arguments

### New Notebooks

**`notebooks/03_model_experiments_v2.ipynb`** - Comprehensive experimental pipeline
- Feature analysis (correlation-based ranking)
- Data preparation with chronological splits and weighting
- Comparison of different weighting strategies
- **Optuna tuning for both Poisson and Tree models** (30 trials each)
- Baseline model comparison
- Feature importance extraction
- Error analysis and sample predictions
- **Focuses on exact score accuracy as primary metric**

---

## Quick Start

### 1. Run Experiments in Notebook (10-15 minutes)

```bash
cd notebooks
# Open 03_model_experiments_v2.ipynb in Jupyter or VS Code
# Run all cells from top to bottom
```

This will:
- ✅ Run Optuna tuning for 30 trials each (Poisson and Tree)
- ✅ Identify best hyperparameters
- ✅ Show which model (Poisson vs Tree) is better
- ✅ Extract top 20 important features
- ✅ Compare performance on exact score accuracy, result accuracy, and goal MAE
- ✅ Print exact error analysis by competition

**Key output to save:**
```
Best Model: TreeOptuna(depth=18, n_est=150)
Exact Score Accuracy: 22.5%
Result Accuracy: 58.3%
Goal MAE: 0.892
```

### 2. Train Production Model (2-3 minutes)

After identifying best parameters from notebook:

```bash
# Train with optimized Tree model and tournament weighting
python -m src.models.train_production \
  --model-type tree \
  --temporal-decay \
  --evaluate

# Or: Train with Poisson
python -m src.models.train_production \
  --model-type poisson \
  --evaluate

# Or: Train ensemble of both
python -m src.models.train_production \
  --model-type ensemble \
  --evaluate
```

This saves:
- `models/saved/production_model.pkl` - Trained model
- `models/saved/model_config.json` - Config with feature list and hyperparameters

### 3. Make Predictions

```python
from src.models.train import load_model
import json

# Load model and config
model = load_model('models/saved/production_model.pkl')
with open('models/saved/model_config.json') as f:
    config = json.load(f)

# Load new data
from src.models.base import load_model_dataset
df = load_model_dataset()

# Prepare features
feature_cols = config['feature_columns']
X = df[feature_cols].fillna(0)

# Predict
y_pred = model.predict(X)
y_pred = np.clip(y_pred, 0, None)

# Convert to scores
predicted_scores = np.round(y_pred).astype(int)
```

---

## Understanding Tournament Weights

Your dataset has competitions with very different importance levels:

```python
from src.models.weighting import COMPETITION_WEIGHTS

# Current weights
COMPETITION_WEIGHTS = {
    'FIFA World Cup': 5.0,           # Most important
    'World Cup qualifier': 3.0,
    'European Championship': 4.0,
    'Copa America': 3.5,
    'African Nations Cup': 3.0,
    'Friendly': 1.0,                # Least important
}
```

**What this means:**
- A World Cup match is worth **5x a friendly match** for training
- A World Cup qualifier is worth **3x a friendly match**
- Friendlies have base weight of 1.0x

**In training:**
- 1000 friendlies (~1000 weight units) = 200 World Cup matches (~1000 weight units)
- Major tournaments drive model learning more strongly

**Apply weights with:**
```python
from src.models.weighting import apply_competition_weights

weights = apply_competition_weights(df_train)
model.fit(X_train, y_train, sample_weight=weights)
```

---

## Understanding Optuna Tuning

Optuna automatically searches for best hyperparameters by maximizing **exact score accuracy**.

### For Poisson Model:

Searches over:
- `alpha`: [0.01, 100.0] (regularization strength)
- `max_iter`: [500, 2000] (solver iterations)

Each trial:
1. Creates a PoissonGoalModel with suggested parameters
2. Trains on training set
3. Evaluates on validation set
4. Returns exact score accuracy
5. Optuna decides next parameters to try based on previous results

**Best practices:**
- ✅ Stops when validation accuracy plateaus
- ✅ Uses median-based pruning to skip bad trials early
- ✅ Returns the model that achieved best validation accuracy

### For Tree Model:

Searches over:
- `n_estimators`: [50, 300] (number of trees)
- `max_depth`: [5, 30] (tree depth)

Same process, optimizes for exact score accuracy.

---

## Feature Selection Process

The notebook extracts **tree feature importance** which tells you:
- Which features the model actually uses the most
- Ordered by importance

**Top features typically include:**
- `elo_diff` - Elo rating difference (team strength)
- `form_diff_last5` - Recent form difference
- `market_value_diff` - Squad quality difference
- `opponent_strength_diff_last5` - Recent opponent strength
- `tournament_goal_diff_diff` - Current tournament performance

**For production training:**
1. Run notebook to extract top 20 features
2. Consider using only top 20 (reduces overfitting, speeds up training)
3. Or use all features (may be more accurate but slower)

**Test both approaches:**
```python
# Approach A: All features
X_train = df_train[feature_cols].fillna(0)
model.fit(X_train, y_train)

# Approach B: Top 20 only
top_20_features = ['elo_diff', 'form_diff_last5', ...]  # from notebook
X_train = df_train[top_20_features].fillna(0)
model.fit(X_train, y_train)
```

---

## Why Exact Score Accuracy Is Low (18-22%)

Predicting exact scores is **very hard** because:

1. **Randomness in football** - Even equally matched teams get different scores
2. **Limited training data** - 21,539 matches means each score combination seen few times
3. **Feature quality** - Your features capture team strength, not individual match randomness
4. **Poisson limitation** - Real goal distributions aren't perfectly Poisson

**Comparison to other metrics:**
- Exact score: ~20% (what we optimize for)
- Result (win/draw/loss): ~55-60% (much easier - pick the stronger team)
- Goal MAE: ~0.9 goals (actually quite good)

**Why optimize for exact score despite low accuracy:**
- ✅ Most valuable for betting and tournaments
- ✅ Implies good calibration of expected goals
- ✅ Harder metric forces better generalization
- ✅ If you predict exact scores well, other metrics follow

---

## Weighting in Your Training

### Default: Competition Weights Only

```python
weights = apply_competition_weights(df_train)
# Simple: multiply sample weight by tournament importance
```

### Advanced: Combined Weights (Competition + Temporal Decay)

```python
weights = apply_combined_weighting(
    df_train,
    apply_decay=True,          # Weight older data less
    decay_rate=0.95,           # 5% decay per year
    reference_year=2024,       # Decay from this year
    competition_weight=0.6,    # 60% weight to competition importance
    temporal_weight=0.4,       # 40% weight to recency
)
```

This means:
- **Recent major tournaments** get highest weight (5.0 * 0.95^0 = 5.0)
- **Old friendlies** get lowest weight (1.0 * 0.95^20 = 0.36)
- Encourages model to adapt to recent playstyles

---

## Interpretation: What Each Metric Means

After experiments, you'll see a table like:

| Model | Exact Acc | Result Acc | Goal MAE |
|-------|-----------|------------|----------|
| TreeOptuna | 22.1% | 58.3% | 0.892 |
| EloBaseline | 15.2% | 52.1% | 1.234 |

**Exact Accuracy 22.1%:**
- Out of 4,307 test matches, model predicted exact score in 950 cases
- Example: Predicted 1-0, actual was 1-0 ✓

**Result Accuracy 58.3%:**
- Out of 4,307 test matches, model got win/draw/loss right in 2,509 cases
- Example: Predicted home win, actual was home win ✓
- Example: Predicted draw, but actual was away win ✗

**Goal MAE 0.892:**
- Average prediction is 0.892 goals off in each team
- If predict 2.0-1.5, actual might be 1.2-2.3

---

## Next Steps After First Run

### Week 1: Initial Experiments (what you're doing now)
- ✅ Run notebook 03_model_experiments_v2.ipynb
- ✅ Identify best model and hyperparameters
- ✅ Extract top features

### Week 2: Production Training
- Train production model with best hyperparameters
- Save model and config
- Test on real 2024-2025 international matches
- Compare predictions to actual results

### Week 3: Refinement
- Analyze prediction errors by competition
- Test feature selection: all vs. top 20
- Try different weighting strategies (with/without temporal decay)
- Compare Poisson vs Tree in production

### Week 4: Ensemble & Production Deployment
- Train ensemble combining best Poisson + best Tree
- Evaluate on regional tournaments (Copa America, AFCON)
- Fine-tune weights between models
- Deploy for 2026 World Cup predictions

---

## Troubleshooting

### "Optuna is slow (taking >5 minutes per model)"
- Reduce n_trials (currently 30) to 10-15
- Increase timeout to timeout=600 seconds
- Optuna will find good solutions faster with fewer trials

### "Exact score accuracy is too low"
- Expected: 18-22% is actually reasonable
- Try with top 20 features only (may reduce overfitting)
- Try ensemble (often better than single model)
- Try different weighting strategies

### "Tree model takes too long to train"
- Reduce n_estimators (currently 150) to 100
- Increase max_depth constraints (currently 5-30) to 5-20
- Use fewer features (top 20 instead of all 34)

### "Model weights aren't being applied"
- Ensure you call `model.fit(X, y, sample_weight=weights)`
- Not all model types support sample_weight (check: Poisson ✓, Tree ✓)

---

## Files Reference

**New/Modified:**
- `src/models/optuna_tuning.py` - Hyperparameter tuning
- `src/models/feature_analysis.py` - Feature importance
- `src/models/weighting.py` - Sample weighting
- `src/models/train_production.py` - Production training
- `notebooks/03_model_experiments_v2.ipynb` - Full experimental pipeline

**Existing (still used):**
- `src/models/base.py` - Data loading and utilities
- `src/models/poisson_model.py` - Poisson model
- `src/models/tree_model.py` - Tree model
- `src/models/baseline.py` - Baseline models
- `src/models/train.py` - Generic training

---

## Questions?

- **"Why Optuna?"** - Systematic hyperparameter search beats manual tuning
- **"Can I use other models?"** - Yes, follow the pattern in optuna_tuning.py to add XGBoost, LightGBM, etc.
- **"Should I use all features?"** - Try both: all features vs top 20 from feature importance
- **"How to handle new data?"** - Retrain with all historical data + new matches, save new model

---

## Commands Quick Reference

```bash
# Run experiments
cd notebooks && jupyter notebook 03_model_experiments_v2.ipynb

# Train production model with best params
python -m src.models.train_production --model-type tree --evaluate

# Or with temporal decay
python -m src.models.train_production --model-type tree --temporal-decay --evaluate

# Or ensemble
python -m src.models.train_production --model-type ensemble --evaluate
```

Good luck with your experiments! 🎯
