# Machine Learning Projects

A collection of machine learning and data analysis projects covering regression modeling and exploratory data analysis. Each project follows a complete workflow: data cleaning, feature engineering, exploratory analysis, modeling, and evaluation.

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

## Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Libraries:** pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, wordcloud

## Datasets

The datasets are not included in this repository (CSV files are gitignored). They can be downloaded from public sources:

- **TMDB 5000 Movies Dataset** -- available on Kaggle
- **NYC Airbnb Open Data (2019)** -- available on Kaggle

Place the CSV files in the corresponding project folder before running the notebooks.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Hanlzheng/machine-learning-practice-projects.git
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn wordcloud jupyter
   ```
3. Download the datasets (see above) and place them in the matching project folders.
4. Launch Jupyter and open any notebook:
   ```bash
   jupyter notebook
   ```