# Kaggle Playground Series

My solutions to competitions from the [Kaggle Playground Series](https://www.kaggle.com/competitions?hostSegmentIdFilter=8) —
a monthly series of approachable, well-structured machine learning challenges.
Each subfolder is one competition, with its own notebook(s) and write-up.

Across these projects I practice a consistent, end-to-end tabular-ML workflow:
exploratory data analysis, feature engineering, leakage-free cross-validation,
hyperparameter tuning, and model ensembling — with all decisions driven by
cross-validation scores rather than the public leaderboard.

## Competitions

| Folder | Competition | Task | Metric |
|--------|-------------|------|--------|
| `s6e5` | Predicting F1 Pit Stops | Binary classification | ROC-AUC |
| `s5e7` | _(add description)_ | _(add)_ | _(add)_ |
| `s5e6` | _(add description)_ | _(add)_ | _(add)_ |

> Fill in the rows above for each competition. Add a new row whenever you add a folder.

## Highlights

**`s6e5` — Predicting F1 Pit Stops**
Predict whether an F1 car pits on the next lap. Full workflow: EDA (found that
each race's lap sequence is heavily down-sampled, so rows are treated as
independent snapshots), feature engineering validated against CV, leakage-free
target encoding inside each fold, Optuna-tuned LightGBM / XGBoost / CatBoost,
and a tuned LGBM + XGBoost blend. Best public score ~0.9496.

_(Add a short highlight paragraph for each competition as you go.)_

## What I Practice Here

- **Reliable validation first** — verified, leakage-free cross-validation;
  decisions are made on CV, not the public leaderboard.
- **Controlled experiments** — change one thing at a time, keep it only if CV
  confirms it helps. Most feature ideas don't help, and that's expected.
- **Honest model selection** — try multiple models, but let CV decide which
  ones make the final ensemble.

## Tools

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · CatBoost · Optuna · matplotlib · seaborn

## Structure

```
playground/
├── s5e6/        # competition notebook(s)
├── s5e7/        # competition notebook(s)
└── s6e5/        # competition notebook(s)
```
