import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed_data", "la_liga_data.csv")

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["target_binary"] = df["FTR"].apply(lambda x: 1 if x == "H" else 0)

FEATURES = [
    "p_home", "p_draw", "p_away",
    "elo_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "xpts_diff", "ppda_diff", "sot_diff",
    "home_xg_avg", "away_xg_avg", "home_xga_avg", "away_xga_avg",
]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

X_train = train[FEATURES]
X_test  = test[FEATURES]
y_train = train["target_binary"]
y_test  = test["target_binary"]

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = LogisticRegression(max_iter=10000, C=0.05, solver="lbfgs")
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("=== Binary Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

coef_df = pd.DataFrame({
    "feature":          FEATURES,
    "coef_home_vs_not": model.coef_[0],
}).sort_values("coef_home_vs_not", ascending=False)

print("Top Positive Features (increase Home win probability):")
print(coef_df.head(10))

print("\nTop Negative Features (decrease Home win probability):")
print(coef_df.tail(10))

# ---------------------------------------------------------------------------
# Bookmaker baseline
# ---------------------------------------------------------------------------

odds_home = (X_test["p_home"] > X_test[["p_draw", "p_away"]].max(axis=1)).astype(int)
print("\nBookmaker Accuracy (Home vs Not Home):", accuracy_score(y_test, odds_home))
