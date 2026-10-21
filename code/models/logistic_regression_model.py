import os
import pandas as pd
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_data", "la_liga_data.csv")

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)
df["Season"] = df["Season"].astype(str).str[:4].astype(int)

FEATURES = [
    "p_home", "p_draw", "p_away",
    "elo_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "xpts_diff", "ppda_diff", "sot_diff",
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
    # "elo_form_interaction",
]

mapping = {"H": 0, "D": 1, "A": 2}
df["target"] = df["FTR"].map(mapping)

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

np.random.seed(42)

model_df = df[FEATURES + ["target", "Season"]].dropna()

TRAIN_SEASONS_MAX = 2023
TEST_SEASON       = 2024

train = model_df[model_df["Season"] <= TRAIN_SEASONS_MAX]
test  = model_df[model_df["Season"] == TEST_SEASON]

X_train = train[FEATURES]
y_train = train["target"]
X_test  = test[FEATURES]
y_test  = test["target"]

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Number of training matches:", len(train))
print("Number of test matches:", len(test))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = LogisticRegression(
    max_iter=10000,
    C=0.05,
    multi_class="multinomial",
    solver="lbfgs",
)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
ll = log_loss(y_test, y_prob)
print("Log Loss:", ll)
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

coef_df = pd.DataFrame({
    "feature":      FEATURES,
    "coef_class_0": model.coef_[0],
    "coef_class_1": model.coef_[1],
    "coef_class_2": model.coef_[2],
})
print(coef_df.sort_values("coef_class_0", ascending=False))

# ---------------------------------------------------------------------------
# Sample predictions
# ---------------------------------------------------------------------------

sample        = X_test.iloc[0:5].copy()
sample_scaled = scaler.transform(sample)
probs         = model.predict_proba(sample_scaled)

result = sample.copy()
result["pred"]      = model.predict(sample_scaled)
result["prob_home"] = probs[:, 0]
result["prob_draw"] = probs[:, 1]
result["prob_away"] = probs[:, 2]
print(result)

# ---------------------------------------------------------------------------
# Per-match contribution breakdown
# ---------------------------------------------------------------------------

i             = 17
x             = X_test_s[i]
coefs         = model.coef_
contrib_home  = x * coefs[0]
contrib_draw  = x * coefs[1]
contrib_away  = x * coefs[2]

contrib_df = pd.DataFrame({
    "feature":      FEATURES,
    "home_contrib": contrib_home,
    "draw_contrib": contrib_draw,
    "away_contrib": contrib_away,
})
print(contrib_df.sort_values("home_contrib", ascending=False))

print(X_test.corrwith(y_test))

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

explainer   = shap.LinearExplainer(model, X_train_s)
shap_values = explainer.shap_values(X_test_s)
shap.summary_plot(shap_values, X_test, feature_names=FEATURES)

# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------

df_test = X_test.copy()
df_test["true"]       = y_test
df_test["pred"]       = y_pred
probs                 = model.predict_proba(X_test_s)
df_test["confidence"] = probs.max(axis=1)
bins                  = pd.qcut(df_test["confidence"], 5)
print(df_test.groupby(bins).apply(lambda x: (x["true"] == x["pred"]).mean()))

# ---------------------------------------------------------------------------
# Odds comparison
# ---------------------------------------------------------------------------

odds_pred    = np.argmax(X_test[["p_home", "p_draw", "p_away"]].values, axis=1)
model_correct = (y_pred == y_test)
odds_correct  = (odds_pred == y_test)

comparison = pd.DataFrame({"model": model_correct, "odds": odds_correct})
print("Model better:", ((comparison["model"] == 1) & (comparison["odds"] == 0)).sum())
print("Odds better:", ((comparison["model"] == 0) & (comparison["odds"] == 1)).sum())

df_test["odds_pred"]   = odds_pred
disagreements          = df_test[df_test["pred"] != df_test["odds_pred"]]
print("Disagreement accuracy:",
      (disagreements["true"] == disagreements["pred"]).mean())

print(model_df["Season"].unique()[:10])

# ---------------------------------------------------------------------------
# Pipeline version
# ---------------------------------------------------------------------------

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=0.05,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
    )),
])
lr_pipeline.fit(X_train, y_train)
