# Implementation Checklist

## ✅ Phase 1: Run Initial Experiments (15-20 minutes)

- [ ] Navigate to `notebooks/` folder
- [ ] Open `03_model_experiments_v2.ipynb` in Jupyter/VS Code
- [ ] Run all cells from top to bottom
- [ ] **Key Output to Note:**
  - [ ] Best model name and hyperparameters
  - [ ] Exact score accuracy percentage
  - [ ] Result accuracy percentage
  - [ ] Goal MAE value
  - [ ] Top 20 features list
- [ ] Save the notebook output (screenshot or export)

**Time: 15-20 minutes**

---

## ✅ Phase 2: Interpret Results (5 minutes)

- [ ] Read the "FINAL MODEL COMPARISON" table
- [ ] Identify which model was best:
  - [ ] PoissonOptuna
  - [ ] TreeOptuna
  - [ ] Ensemble (if tested)
- [ ] Note the exact score accuracy (should be 18-24%)
- [ ] Check if better than baselines (should beat AverageGoalsBaseline at 12-15%)

**Time: 5 minutes**

---

## ✅ Phase 3: Train Production Model (5 minutes)

Open terminal and run:

```bash
cd c:/Users/yuval/WC_Predictor

# Option A: If Tree was best
python -m src.models.train_production --model-type tree --evaluate

# Option B: If Poisson was best
python -m src.models.train_production --model-type poisson --evaluate

# Option C: Try ensemble
python -m src.models.train_production --model-type ensemble --evaluate
```

- [ ] Wait for training to complete (~2 minutes)
- [ ] Check output messages - should see:
  - [ ] "Model trained successfully"
  - [ ] "Model saved to models/saved/production_model.pkl"
  - [ ] Evaluation results with accuracy metrics
- [ ] Verify files were created:
  - [ ] `models/saved/production_model.pkl` exists
  - [ ] `models/saved/model_config.json` exists

**Time: 5 minutes**

---

## ✅ Phase 4: Understand Tournament Weights (5 minutes)

- [ ] Open `src/models/weighting.py`
- [ ] Review `COMPETITION_WEIGHTS` dictionary
- [ ] Understand: World Cup = 5.0x, Friendly = 1.0x
- [ ] Consider if you want to adjust any weights
- [ ] If you want to change weights:
  - [ ] Edit the COMPETITION_WEIGHTS dictionary
  - [ ] Re-run Phase 3 (training)
  - [ ] Compare results

**Time: 5 minutes**

---

## ✅ Phase 5: Feature Analysis (5 minutes)

From the notebook output:
- [ ] Write down top 10 features mentioned
- [ ] Look at the bar chart showing feature importance
- [ ] Note which features have highest importance
- [ ] Consider: "Do these make football sense?"
  - [ ] elo_diff (team strength) - YES, makes sense
  - [ ] form_diff_last5 (recent form) - YES, makes sense
  - [ ] market_value_diff (squad quality) - YES, makes sense

**Time: 5 minutes**

---

## ✅ Phase 6: Test Production Model (10 minutes)

Create a test script or Jupyter cell:

```python
import json
import numpy as np
from src.models.train import load_model
from src.models.base import load_model_dataset

# Load trained model
model = load_model('models/saved/production_model.pkl')
with open('models/saved/model_config.json') as f:
    config = json.load(f)

# Load fresh data
df = load_model_dataset()
feature_cols = config['feature_columns']

# Prepare features (last 10 matches)
X_test = df[feature_cols].iloc[-10:].fillna(0)
teams_a = df['team_A'].iloc[-10:].values
teams_b = df['team_B'].iloc[-10:].values

# Make predictions
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, None)

# Display
for i, (team_a, team_b, pred) in enumerate(zip(teams_a, teams_b, y_pred)):
    print(f"{team_a} vs {team_b}: {pred[0]:.1f}-{pred[1]:.1f}")
```

- [ ] Run test script
- [ ] See predictions for recent matches
- [ ] Manually compare to actual results if available
- [ ] Check: Do predictions seem reasonable?

**Time: 10 minutes**

---

## ✅ Phase 7: Document Results (5 minutes)

Create a file `EXPERIMENT_RESULTS.md`:

```markdown
# Experiment Results - [DATE]

## Best Model
- Model Type: [Tree/Poisson/Ensemble]
- Exact Score Accuracy: [XX.X%]
- Result Accuracy: [XX.X%]
- Goal MAE: [X.XXX]

## Hyperparameters
[Copy from notebook output]

## Top 5 Features
1. [Feature]
2. [Feature]
3. [Feature]
4. [Feature]
5. [Feature]

## Key Observations
- [Observation 1]
- [Observation 2]
- [Observation 3]

## Next Steps
- [ ] [Action item]
- [ ] [Action item]
```

- [ ] Create and save this file
- [ ] Keep for reference

**Time: 5 minutes**

---

## ✅ Phase 8: Plan Next Week (5 minutes)

- [ ] Review notebook output weekly
- [ ] Track if accuracy improves/decreases
- [ ] Consider improvements for next iteration:
  - [ ] Try only top 20 features (vs all 34)
  - [ ] Try different tournament weights
  - [ ] Try with temporal decay enabled
  - [ ] Compare Poisson vs Tree on real data
- [ ] Schedule: Run full experiment again in 1 week

**Time: 5 minutes**

---

## 📊 Success Criteria (You'll Know It Worked If...)

### ✅ Minimum Success
- [ ] Notebook runs without errors
- [ ] Exact score accuracy >= 15%
- [ ] Production model trains successfully
- [ ] Can make predictions on new data

### ✅ Good Success
- [ ] Exact score accuracy >= 20%
- [ ] Result accuracy >= 55%
- [ ] Tree model outperforms Poisson
- [ ] Top features match your intuition (elo, form, etc.)

### ✅ Great Success
- [ ] Exact score accuracy >= 22%
- [ ] Result accuracy >= 58%
- [ ] Feature importance extraction works
- [ ] Different weighting strategies show clear differences

---

## 🚨 Troubleshooting Checklist

### Notebook won't run
- [ ] Check you're in `notebooks/` folder
- [ ] Check file path is correct: `03_model_experiments_v2.ipynb`
- [ ] Try: `python -c "from src.models.base import load_model_dataset; print(load_model_dataset().shape)"`

### Optuna tuning takes too long (>15 min)
- [ ] Normal: ~6 minutes for 30+30 trials
- [ ] If longer: Check CPU usage, close other programs
- [ ] Workaround: Reduce n_trials to 15 each in notebook

### Exact score accuracy is very low (< 10%)
- [ ] Check: Are predictions all the same value?
- [ ] Check: Is test set valid (different from train)?
- [ ] Try: Disable weights temporarily, retrain

### "Module not found" errors
- [ ] Check: All imports pass the verification script
- [ ] Try: Restart Python kernel
- [ ] Try: Run from repo root: `python -c "from src.models.optuna_tuning import PoissonTuner"`

### Model file not found after training
- [ ] Check: Does `models/` directory exist?
- [ ] Check: Was training command completed (no interruption)?
- [ ] Check: File permissions allow creation?

---

## 📝 Quick Reference Commands

```bash
# Run experiments
cd notebooks && jupyter notebook 03_model_experiments_v2.ipynb

# Train production model
python -m src.models.train_production --model-type tree --evaluate

# With temporal decay (weights recent higher)
python -m src.models.train_production --model-type tree --temporal-decay --evaluate

# Verify setup
python -c "from src.models.optuna_tuning import PoissonTuner; print('OK')"

# Check data loads
python -c "from src.models.base import load_model_dataset; print(load_model_dataset().shape)"
```

---

## 🎯 Timeline Estimate

| Phase | Time | Status |
|-------|------|--------|
| 1. Run experiments | 15 min | 🔲 |
| 2. Interpret results | 5 min | 🔲 |
| 3. Train production | 5 min | 🔲 |
| 4. Understand weights | 5 min | 🔲 |
| 5. Feature analysis | 5 min | 🔲 |
| 6. Test model | 10 min | 🔲 |
| 7. Document results | 5 min | 🔲 |
| 8. Plan next week | 5 min | 🔲 |
| **Total** | **55 min** | 🔲 |

---

## 💡 Pro Tips

- **Tip 1:** Run notebook with `%%time` magic to see cell execution times
- **Tip 2:** Save notebook outputs before closing to reference later
- **Tip 3:** Keep experiment results in a CSV to track improvement over time
- **Tip 4:** Test predictions on matches you know the actual results for
- **Tip 5:** Don't change too many things at once (one variable at a time)

---

## 📞 Final Notes

- All files are ready to use, no additional setup needed
- Notebook is self-contained, can be shared with team members
- Production script is CLI-friendly, can be automated
- Feature extraction helps guide future feature engineering with your partner

**You're ready to go! Start with Phase 1.** 🚀
