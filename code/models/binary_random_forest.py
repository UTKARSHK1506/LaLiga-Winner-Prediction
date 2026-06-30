import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("=== Random Forest (Binary) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

feat_imp = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("Top Features:")
print(feat_imp.head(10))

print("\nLeast Important Features:")
print(feat_imp.tail(10))
