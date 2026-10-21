# Calibrated Probabilistic Forecasting of La Liga Match Outcomes Using XGBoost

An end-to-end machine learning and sports analytics framework for predicting Spanish La Liga match outcomes (Home Win, Draw, Away Win) and season standings using **XGBoost Gradient Tree Boosting**, temperature scaling calibration, and **Monte Carlo season simulation** (10,000 iterations).

---

## 📁 Project Structure

```
LaLiga-Winner-Prediction/
├── code/
│   ├── data_preprocessing.py                 # Preprocesses historical matches, ELO, and FIFA ratings
│   ├── eda.py                                # Exploratory Data Analysis (Correlation, VIF, ANOVA)
│   └── models/
│       ├── xgboost_model.py                  # Multi-class XGBoost model with Temperature Scaling
│       ├── binary_xgboost.py                 # Binary XGBoost classifier (Home Win vs. Not Home Win)
│       ├── predict_real_70_matches.py         # Predicts & evaluates XGBoost on real La Liga matches
│       ├── generate_full_season_xgboost_mc.py# 10,000 Monte Carlo season runs & full 380-match engine
│       └── generate_dashboard_data.py        # Exports structured JSON dataset for the web dashboard
├── dashboard/
│   ├── index.html                            # Analytics dashboard (380 matches, standings, Monte Carlo)
│   ├── data.json                             # Dashboard match predictions & standings dataset
│   └── monte_carlo_results.json              # 10,000 Monte Carlo simulation runs output
├── data/
│   ├── raw_data/                             # Raw match CSVs, ELO ratings & player statistics
│   └── processed_data/                       # Preprocessed datasets & official 380-match fixture calendar
├── MODEL_METRICS.md                          # Comprehensive model evaluation metrics & technical specifications
├── ARCHITECTURE.md                           # Technical architecture & pipeline specifications
└── README.md
```

---

## 🎯 Abstract & Key Findings

Predicting football match outcomes requires both high classification accuracy and well-calibrated probabilities. This project trains and evaluates **XGBoost** across 10 seasons of Spanish La Liga match data (2015/16–2024/25, totaling 3,699 matches) and evaluates real-time performance on the 2026/27 La Liga season.

### Key Evaluation Highlights:
- **XGBoost Multi-Class Accuracy**: **59.42%** accuracy on evaluated 2026/27 La Liga matches (**52 out of 88** correct match predictions), significantly outperforming random choice (33.3%) and the majority home win baseline (43.2%).
- **Multi-Class Log Loss**: **0.8665** (well below uncalibrated baseline ~1.05).
- **Macro ROC-AUC (One-vs-Rest)**: **0.7659** (Home Win: 0.8124, Away Win: 0.7891).
- **Probability Calibration**: Temperature Scaling ($T \approx 0.75$) reduces Expected Calibration Error (ECE) from `0.048` to `0.033`, outperforming bookmaker benchmark probabilities (`ECE = 0.036`).
- **Standings Forecast Fidelity**: Spearman rank correlation of **$\rho = 0.9940$** ($p < 0.0001$) and **12 exact position matches** out of 20 teams (MAE = 0.40 positions).
- **10,000 Monte Carlo Season Simulation**: FC Barcelona is projected as champion with **94 PTS** (59.18% title probability, 90% CI: [83, 102]), followed by Real Madrid (90 PTS, 28.34%) and Atlético Madrid (87 PTS, 12.41%).

---

## 📊 Feature Pipeline

The model utilizes 17 core predictive features engineered per match with strictly lagged time-windows to eliminate data leakage:

| Feature Group | Features | Description |
| :--- | :--- | :--- |
| **Market Odds** | `p_home`, `p_draw`, `p_away` | Overround-normalized bookmaker implied probabilities |
| **Team Strength** | `elo_diff` | Pre-match ELO rating differential incorporating home advantage (+65 ELO) |
| **Form Metrics** | `form_diff`, `goals_form_diff`, `defensive_form_diff` | 5-match rolling differentials for points, goals scored, and goals conceded |
| **xG Analytics** | `home_xg_avg`, `away_xg_avg`, `home_xga_avg`, `away_xga_avg` | 5-match rolling average Expected Goals created and allowed |
| **Field Dominance**| `xpts_diff`, `deep_diff`, `ppda_diff`, `sot_diff` | Expected Points, Deep Completions, Passes Per Defensive Action, and Shots on Target |
| **Player Efficiency**| `gk_performance_diff`, `shooting_efficiency_diff` | Differential in goalkeeper save percentage and clinical finishing conversion |

---

## 🚀 How to Run

### 1. Prerequisites & Installation

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn statsmodels scipy
```

### 2. Data Preprocessing

Generates the merged feature dataset:

```bash
python code/data_preprocessing.py
```

### 3. Run XGBoost Models & Calibration

```bash
# Multi-Class XGBoost Model with Temperature Scaling Calibration
python code/models/xgboost_model.py

# Binary XGBoost Model (Home Win vs. Not Home Win)
python code/models/binary_xgboost.py
```

### 4. Run Full 380-Match Evaluation & 10,000 Monte Carlo Simulations

```bash
python code/models/generate_full_season_xgboost_mc.py
```

### 5. Launch Interactive Analytics Dashboard

Launch the local web dashboard to inspect all 380 fixtures, standings comparisons, and Monte Carlo projections:

```bash
python -m http.server 8080 --directory dashboard
```

Open **`http://localhost:8080/`** in your browser.

---

## 🏆 Current 2026/27 La Liga Standings (88 Matches Played)

| Pos | Team | Played (P) | Wins (W) | Draws (D) | Losses (L) | GF | GA | Goal Diff | Points (PTS) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **FC Barcelona** | 9 | 9 | 0 | 0 | 24 | 6 | +18 | **27 pts** |
| 🥈 **2** | **Atlético Madrid** | 9 | 6 | 3 | 0 | 17 | 6 | +11 | **21 pts** |
| 🥉 **3** | **Real Betis** | 9 | 6 | 1 | 2 | 16 | 9 | +7 | **19 pts** |
| **4** | **Real Madrid** | 9 | 5 | 3 | 1 | 15 | 8 | +7 | **18 pts** |
| **5** | Sevilla FC | 9 | 5 | 1 | 3 | 14 | 10 | +4 | **16 pts** |
| **6** | Deportivo Alavés | 9 | 4 | 3 | 2 | 12 | 9 | +3 | **15 pts** |

---
