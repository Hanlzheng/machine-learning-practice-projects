# Machine Learning Projects

A collection of machine learning and data analysis projects covering regression modeling, exploratory data analysis, and competitive machine learning. Each project follows a complete workflow: data cleaning, feature engineering, exploratory analysis, modeling, and evaluation.

## Projects

### 1. Movie Revenue Prediction

Predicting box office revenue using the TMDB 5000 Movies Dataset.

- **Goal:** Predict a film's revenue from its production and metadata features.
- **Workflow:**
  - Cleaned the data by removing entries with missing budget/revenue and filtering revenue outliers using the IQR method
  - Parsed JSON-formatted columns (genres, keywords, cast, crew, production companies) and engineered numeric features such as genre count, keyword count, and director
  - Performed EDA on release trends, genre trends over time, revenue distribution, and feature correlations
  - Trained and compared three regression models using a train / validation / test split, with `StandardScaler` fit only on training data to prevent data leakage
- **Models:** Linear Regression, Random Forest, XGBoost
- **Results (test set):**

  | Model | RMSE | R-squared |
  |-------|------|-----------|
  | Linear Regression | $81.9M | 0.532 |
  | Random Forest | $80.0M | 0.554 |
  | XGBoost | $80.4M | 0.551 |

  Random Forest performed best, explaining roughly 55% of revenue variance. `vote_count`, `budget`, and `popularity` were the strongest predictors -- suggesting financial success is driven mainly by budget and audience engagement rather than release timing or location.
- **Tech:** Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- View notebook: [movie-revenue-prediction/](./movie-revenue-prediction/)

### 2. Energy Usage Prediction

Predicting daily household energy consumption from weather data.

- **Goal:** Predict daily energy usage (kWh) using weather features.
- **Workflow:**
  - Merged energy and weather datasets on date, converting timestamps to datetime
  - Aggregated 30-minute energy readings into daily totals and averaged hourly weather readings per day
  - Used a time-based train/test split (trained on Jan-Nov, tested on December)
  - Trained a Linear Regression model with standardized features
- **Tech:** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
- View notebook: [energy-usage-prediction/](./energy-usage-prediction/)

### 3. NYC Airbnb Exploratory Data Analysis

Exploratory analysis of the 2019 New York City Airbnb listings dataset (~49,000 listings).

- **Goal:** Uncover pricing and availability patterns across NYC neighborhoods.
- **Workflow:**
  - Cleaned the dataset by handling missing values and removing invalid listings
  - Analyzed price distributions, top and bottom neighborhoods by average price, and listing patterns
  - Visualized findings with charts and word clouds
- **Tech:** Python, pandas, NumPy, matplotlib, seaborn, wordcloud
- View notebook: [nyc-airbnb-eda/](./nyc-airbnb-eda/)

### 4. Kaggle Competitions

Solutions to competitions from the Kaggle Playground Series, applying a full competitive machine learning workflow.

- **Goal:** Build accurate, well-validated models for monthly Kaggle Playground competitions.
- **Workflow:**
  - Performed EDA to understand data structure, target balance, and feature/target relationships
  - Engineered and validated features against cross-validation, keeping only those that improved the score
  - Used k-fold `StratifiedKFold` cross-validation with leakage-free target encoding computed inside each fold
  - Tuned hyperparameters with Optuna and ensembled multiple gradient-boosting models
  - Made all modeling decisions based on cross-validation scores rather than the public leaderboard
- **Featured competition -- Predicting F1 Pit Stops (Playground Series S6E5):**
  predict whether a Formula 1 car pits on the next lap (binary classification, ROC-AUC).
  Found that each race's lap sequence is heavily down-sampled, so rows are treated
  as independent snapshots; final solution blends Optuna-tuned LightGBM and XGBoost
  models. Best public score ~0.9496.
- **Models:** LightGBM, XGBoost, CatBoost
- **Tech:** Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, matplotlib, seaborn
- View competitions: [kaggle/](./kaggle/)

## Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Libraries:** pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, matplotlib, seaborn, wordcloud

## Datasets

The datasets are not included in this repository (CSV files are gitignored). They can be downloaded from public sources:

- **TMDB 5000 Movies Dataset** -- available on Kaggle
- **NYC Airbnb Open Data (2019)** -- available on Kaggle
- **Kaggle Playground Series** -- competition data available on each competition page

Place the CSV files in the corresponding project folder before running the notebooks.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Hanlzheng/machine-learning-practice-projects.git
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna matplotlib seaborn wordcloud jupyter
   ```
3. Download the datasets (see above) and place them in the matching project folders.
4. Launch Jupyter and open any notebook:
   ```bash
   jupyter notebook
   ```
