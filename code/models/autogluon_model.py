import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, classification_report, log_loss

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed_data", "la_liga_data.csv")

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)

FEATURES = [
    "p_home", "p_draw", "p_away",
    "elo_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "xpts_diff", "ppda_diff", "sot_diff",
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
]

train_df = df[df["matchDate"].dt.year < 2024]
test_df  = df[df["matchDate"].dt.year == 2024]

train_data = train_df[FEATURES + ["FTR"]].dropna()
test_data  = test_df[FEATURES + ["FTR"]].dropna()

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# ---------------------------------------------------------------------------
# Final model — Experiment 10: best_v150, 1200s, f1_macro, balanced weights
# Achieved: Accuracy=0.6430, F1_Draw=0.451, Draw Recall=0.469
# ---------------------------------------------------------------------------

predictor = TabularPredictor(
    label="FTR",
    eval_metric="f1_macro",
    problem_type="multiclass",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ag_final"),
).fit(
    train_data,
    presets="best_v150",
    time_limit=1200,
    ag_args_fit={"class_weights": "balanced"},
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

y_true = test_data["FTR"]
y_pred = predictor.predict(test_data)
y_prob = predictor.predict_proba(test_data)

acc = accuracy_score(y_true, y_pred)

try:
    prob_array = y_prob[["A", "D", "H"]].values
    ll = log_loss(y_true, prob_array, labels=["A", "D", "H"])
except Exception:
    ll = None

rep = classification_report(y_true, y_pred, output_dict=True)

print(f"\n===== AUTOGLUON RESULTS =====")
print(f"Accuracy:    {acc:.4f}")
print(f"Log Loss:    {ll:.4f}" if ll else "Log Loss:   N/A")
print(f"F1 Home:     {rep.get('H', {}).get('f1-score', 0):.3f}")
print(f"F1 Draw:     {rep.get('D', {}).get('f1-score', 0):.3f}")
print(f"F1 Away:     {rep.get('A', {}).get('f1-score', 0):.3f}")
print(f"Draw Recall: {rep.get('D', {}).get('recall', 0):.3f}")
print(f"Best Model:  {predictor.model_best}")
print()
print(classification_report(y_true, y_pred))

# ---------------------------------------------------------------------------
# Leaderboard & feature importance
# ---------------------------------------------------------------------------

leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

fi = predictor.feature_importance(test_data)
print(fi)