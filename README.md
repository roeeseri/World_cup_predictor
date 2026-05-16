# World Cup Score Predictor

**Description:** A lightweight, modular Python project for forecasting World Cup match scores from historical data and team context.

**Goal:** Provide a clean baseline that supports multiple model families (statistical, tree-based, and ensembling), plus evaluation and backtesting to compare approaches.

This is a **dynamic supervised prediction system**, not reinforcement learning. It uses historical match outcomes and evolving team state to update feature inputs over time.

## Planned data sources
- FIFA match results datasets (historical tournaments and qualifiers)
- Public team rankings and ratings (Elo-style sources)
- Team-level pre-tournament indicators (e.g., squad value, recent form)

## Planned features
- Dynamic team state updates (points, goal difference, wins/draws/losses)
- Pre-tournament priors blended with match-by-match updates
- Model zoo: Poisson regression, tree-based models, and ensembles
- Chronological backtesting and evaluation reports

## Planned metrics
- Goal MAE / RMSE
- Exact score accuracy
- Result (win/draw/loss) accuracy
- Goal difference MAE
- Winner-aware error

## Basic run instructions
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Start the Streamlit placeholder app:
   ```bash
   streamlit run src\app\streamlit_app.py
   ```
