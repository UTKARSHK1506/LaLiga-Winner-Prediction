# Calibrated Probabilistic Forecasting of La Liga Match Outcomes Using XGBoost

An end-to-end machine learning and sports analytics framework for predicting Spanish La Liga match outcomes (Home Win, Draw, Away Win) using **XGBoost Gradient Boosting** and advanced feature engineering (Expected Goals, ELO ratings, rolling form, and bookmaker odds).

---

## 📁 Project Structure

```
LaLiga-Winner-Prediction/
├── code/
│   ├── data_preprocessing.py           # Preprocesses historical matches, ELO, and FIFA ratings
│   ├── eda.py                          # Exploratory Data Analysis (Correlation, VIF, ANOVA)
│   └── models/
│       ├── xgboost_model.py            # Multi-class XGBoost model with Temperature Scaling
│       ├── binary_xgboost.py           # Binary XGBoost classifier (Home Win vs. Not Home Win)
│       ├── predict_real_70_matches.py   # Predicts & evaluates XGBoost on 2026/27 La Liga matches
│       └── generate_dashboard_data.py  # Exports clean JSON dataset for the web dashboard
├── dashboard/
│   ├── index.html                      # Clean, professional football analytics dashboard
│   └── data.json                       # Dashboard match predictions & standings dataset
├── data/
│   ├── raw_data/                       # Raw match CSVs, ELO ratings & FIFA datasets
│   └── processed_data/                 # Preprocessed dataset & 2026/27 evaluation outputs
└── README.md
```

---

## 🎯 Abstract & Key Findings

Predicting football match outcomes requires both high classification accuracy and well-calibrated probabilities. This project trains and evaluates **XGBoost** on 10 seasons of Spanish La Liga match data (2015/16–2024/25, totaling 3,699 matches) and tests the model on the 2026/27 La Liga season.

### Key Highlights:
- **XGBoost Multi-Class Accuracy**: **57.97%** accuracy on real 2026/27 La Liga matches (40 out of 69 correct match predictions).
- **Binary Classification (Home Win vs. Not Home Win)**: **73.92% Accuracy** and **0.800 ROC-AUC**.
- **Probability Calibration**: Temperature Scaling ($T \approx 0.75$) reduces Expected Calibration Error (ECE) from `0.048` to `0.033`, outperforming bookmaker benchmark probabilities (`ECE = 0.036`).
- **High Draw Detection**: Standalone XGBoost achieves **47.1% draw recall**, outperforming linear and uncalibrated tree ensembles.

---

## 📊 Feature Pipeline

The model utilizes 15 core predictive features engineered per match:

| Feature Group | Features | Description |
| :--- | :--- | :--- |
| **Market Odds** | `p_home`, `p_draw`, `p_away` | Overround-normalized bookmaker implied probabilities |
| **Team Strength** | `elo_diff` | Pre-match ELO rating differential (`Home ELO - Away ELO`) |
| **Form Metrics** | `form_diff`, `goals_form_diff`, `defensive_form_diff` | 5-match rolling differentials for points, goals scored, and goals conceded |
| **xG Analytics** | `home_xg_avg`, `away_xg_avg`, `home_xga_avg`, `away_xga_avg` | 5-match rolling average Expected Goals created and allowed |
| **Field Dominance**| `xpts_diff`, `deep_diff`, `ppda_diff`, `sot_diff` | Expected Points, Deep Completions, Passes Per Defensive Action, and Shots on Target |

---

## 🚀 How to Run

### 1. Prerequisites & Installation

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn statsmodels scipy
```

### 2. Data Preprocessing

Generates the merged feature dataset `data/processed_data/la_liga_data.csv`:

```bash
python code/data_preprocessing.py
```

### 3. Run XGBoost Models

```bash
# Multi-Class XGBoost Model with Calibration
python code/models/xgboost_model.py

# Binary XGBoost Model (Home Win vs. Not Home Win)
python code/models/binary_xgboost.py
```

### 4. Evaluate 2026/27 Season Matches

Predict and evaluate match outcomes for the current 2026/27 La Liga season:

```bash
python code/models/predict_real_70_matches.py
```

### 5. Interactive Analytics Dashboard

Generate dashboard data and launch the local HTTP web server:

```bash
python code/models/generate_dashboard_data.py
python -m http.server 8080 --directory dashboard
```

Open **`http://localhost:8080/`** in your browser to inspect match-by-match probabilities and real La Liga standings.

---

## 🏆 2026/27 La Liga Standings 

| Pos | Team | Played | Wins | Draws | Losses | Goal Diff | Points |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **FC Barcelona** | 7 | 7 | 0 | 0 | +24 | **21 pts** |
| 🥈 **2** | **Atlético Madrid** | 7 | 5 | 1 | 1 | +9 | **16 pts** |
| 🥉 **3** | **Real Betis** | 7 | 5 | 1 | 1 | +2 | **16 pts** |
| **4** | **Real Madrid** | 7 | 5 | 0 | 2 | +10 | **15 pts** |
