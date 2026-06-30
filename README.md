# Calibrated Probabilistic Forecasting of Football Match Outcomes Using Machine Learning and Advanced Feature Engineering

Edelweiss Gevorgyan - American University of Armenia, College of Science and Engineering

---

## Project Structure

```
capstone_project/
├── paper/
│   ├── paper.tex
│   └── paper.pdf
├── code/
│   ├── data_preprocessing.py
│   ├── eda.py
│   └── models/
│       ├── logistic_regression_model.py
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       ├── autogluon_model.py
│       ├── binary_logistic_regression.py
│       ├── binary_random_forest.py
│       └── binary_xgboost.py
├── data/
│   ├── raw_data/
│   │   ├── la_liga_dataset.csv
│   │   ├── Matches.csv
│   │   └── fifa15.csv - fifa25.csv
│   └── processed_data/
│       └── la_liga_data.csv
└── README.md
```

---

## Abstract

Rather than treating football match outcome prediction purely as a classification problem, this paper presents a machine learning framework with a particular focus on both predictive accuracy and the quality of predicted probabilities. Four models were trained and compared - Logistic Regression, Random Forest, XGBoost, and AutoGluon - on ten seasons of Spanish La Liga data (2015/16-2024/25). AutoGluon achieved the highest classification accuracy of 64.3%. Models were evaluated using Expected Calibration Error, Brier Score, and Ranked Probability Score, benchmarked against bookmaker-implied probabilities (ECE = 0.036). Temperature Scaling reduced XGBoost's ECE from 0.048 to 0.033, outperforming the bookmaker benchmark. A key finding is that classification accuracy and probability calibration are independent properties.

---

## Data

### Raw Data (`data/raw_data/`)

| File | Description |
|------|-------------|
| `la_liga_dataset.csv` | Base match dataset with xG, PPDA, deep completions, xpts, shots, and betting odds - La Liga 2015/16-2024/25 |
| `Matches.csv` | ELO ratings per match (filtered to Spanish Liga SP1) |
| `fifa15.csv` - `fifa25.csv` | FIFA team ratings (Overall, Attack, Midfield, Defence) per season |

### Processed Data (`data/processed_data/`)

| File | Description |
|------|-------------|
| `la_liga_data.csv` | Final merged dataset with all engineered features - 3,699 rows x 85 columns |

---

## How to Run

### 1. Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn statsmodels scipy
```

For AutoGluon (requires sufficient GPU/RAM):
```bash
pip install autogluon.tabular
```

### 2. Preprocessing

Generates `data/processed_data/la_liga_data.csv` from raw sources:

```bash
python code/data_preprocessing.py
```

### 3. EDA

Runs exploratory analysis - result distributions, correlation matrix, VIF, Cramer's V, ANOVA:

```bash
python code/eda.py
```

### 4. Models

```bash
# Multi-class (Home / Draw / Away)
python code/models/logistic_regression_model.py
python code/models/random_forest_model.py
python code/models/xgboost_model.py
python code/models/autogluon_model.py

# Binary (Home Win vs. Not Home Win)
python code/models/binary_logistic_regression.py
python code/models/binary_random_forest.py
python code/models/binary_xgboost.py
```

> AutoGluon requires a machine with sufficient GPU/RAM.

---

## Features

| Group | Features |
|-------|----------|
| Market | `p_home`, `p_draw`, `p_away` - normalised bookmaker-implied probabilities |
| Strength | `elo_diff` - ELO rating difference |
| Form | `form_diff`, `goals_form_diff`, `defensive_form_diff` - 5-game rolling differentials |
| Performance | `deep_diff`, `xpts_diff`, `ppda_diff`, `sot_diff` |
| xG quality | `home_xg_avg`, `away_xg_avg`, `home_xga_avg`, `away_xga_avg` |

---

## Results

### Multi-Class Classification

| Model | Accuracy | Log Loss | F1 Home | F1 Draw | F1 Away | Draw Recall |
|-------|----------|----------|---------|---------|---------|-------------|
| AutoGluon | 0.643 | 0.831 | 0.746 | 0.451 | 0.657 | 0.469 |
| Random Forest | 0.637 | 0.848 | 0.753 | 0.319 | 0.640 | 0.227 |
| Logistic Regression | 0.629 | 0.845 | 0.759 | 0.256 | 0.636 | 0.175 |
| XGBoost | 0.616 | 0.857 | 0.738 | 0.197 | 0.654 | 0.134 |

### Binary Classification (Home Win vs. Not Home Win)

| Model | Accuracy | AUC | F1 Home | F1 Not Home | vs. Bookmaker |
|-------|----------|-----|---------|-------------|---------------|
| Random Forest | 0.753 | 0.811 | 0.78 | 0.72 | +13.8 pp |
| Logistic Regression | 0.747 | 0.817 | 0.78 | 0.71 | +13.2 pp |
| XGBoost | 0.743 | 0.808 | 0.77 | 0.70 | +12.8 pp |

### Calibration Results

| Model | ECE Before | ECE After | Method | vs. Bookmaker |
|-------|-----------|-----------|--------|---------------|
| Logistic Regression | 0.0351 | 0.0351 | None required | Better |
| Random Forest | 0.0453 | 0.0362 | Temperature Scaling (T=0.894) | Matches |
| XGBoost | 0.0480 | 0.0327 | Temperature Scaling (T=0.748) | Better |
| Bookmaker | 0.0361 | - | - | Reference |

### Per-Class ECE Before and After Temperature Scaling

| Model | Class | ECE Before | ECE After |
|-------|-------|-----------|-----------|
| Logistic Regression | Home Win | 0.0347 | 0.0347 |
| | Draw | 0.0157 | 0.0157 |
| | Away Win | 0.0549 | 0.0549 |
| Random Forest | Home Win | 0.0558 | 0.0367 |
| | Draw | 0.0318 | 0.0299 |
| | Away Win | 0.0483 | 0.0421 |
| XGBoost | Home Win | 0.0597 | 0.0453 |
| | Draw | 0.0222 | 0.0257 |
| | Away Win | 0.0621 | 0.0272 |

---

## Key Findings

- **Classification accuracy and probability calibration are independent properties.** A model with higher accuracy can produce less reliable probabilities than a simpler model.
- **Logistic Regression achieved benchmark-level calibration naturally** (ECE = 0.0351) without any post-hoc correction, consistent with its log-loss optimization during training.
- **Temperature Scaling was the most effective post-hoc method**, improving calibration for both Random Forest and XGBoost while preserving classification accuracy.
- **Isotonic Regression performed poorly** on the 380-match test set, consistent with its known requirement for large calibration datasets.
- **Draw prediction was the hardest class** across all models, consistent with the wider football analytics literature.

---

## Experimental Setup

- **Train:** Seasons 2015/16 - 2023/24 (3,319 matches)
- **Test:** Season 2024/25 (380 matches)
- **Split:** Strict temporal split - no future data used during training
- **Calibration:** 5-fold cross-validation to prevent leakage
- **Benchmark:** Bookmaker-implied probabilities (ECE = 0.036)
