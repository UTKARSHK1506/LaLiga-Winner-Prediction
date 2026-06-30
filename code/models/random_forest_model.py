import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, log_loss, classification_report, brier_score_loss

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed_data", "la_liga_data.csv")

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)
df["Season"] = df["Season"].astype(str).str[:4].astype(int)

df["xg_vs_def_diff"] = (
    df["home_xg_avg"] - df["away_xga_avg"]
) - (
    df["away_xg_avg"] - df["home_xga_avg"]
)

FEATURES = [
    "p_home", "p_draw", "p_away",
    "elo_diff", "xpts_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "ppda_diff", "sot_diff",
    "home_xg_avg", "away_xg_avg", "xg_vs_def_diff",
    "home_xga_avg", "away_xga_avg",
]

mapping = {"H": 0, "D": 1, "A": 2}
df["target"] = df["FTR"].map(mapping)
model_df = df[FEATURES + ["target", "Season"]].dropna()

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

TRAIN_SEASONS_MAX = 2023
TEST_SEASON       = 2024

train = model_df[model_df["Season"] <= TRAIN_SEASONS_MAX]
test  = model_df[model_df["Season"] == TEST_SEASON]

X_train = train[FEATURES]
y_train = train["target"]
X_test  = test[FEATURES]
y_test  = test["target"]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=15,
    min_samples_leaf=20,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feat_imp = pd.DataFrame({
    "feature":    FEATURES,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nFeature Importance:")
print(feat_imp)

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

param_grid = {
    "n_estimators":    [100, 200, 300, 400, 500, 600],
    "max_depth":       [5, 7, 10],
    "min_samples_leaf":[5, 10],
    "class_weight":    ["balanced", None],
}

rf_gs     = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_gs, param_grid=param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")

y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
y_pred       = best_rf.predict(X_test)
y_prob_uncal = best_rf.predict_proba(X_test)

print("===== BASE MODEL RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob_uncal))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cal_platt    = CalibratedClassifierCV(best_rf, cv=5, method="sigmoid")
cal_isotonic = CalibratedClassifierCV(best_rf, cv=5, method="isotonic")
cal_platt.fit(X_train, y_train)
cal_isotonic.fit(X_train, y_train)
y_prob_platt    = cal_platt.predict_proba(X_test)
y_prob_isotonic = cal_isotonic.predict_proba(X_test)


def multiclass_brier(y_true, probs, classes=[0, 1, 2]):
    y_bin = label_binarize(y_true, classes=classes)
    return np.mean([brier_score_loss(y_bin[:, i], probs[:, i]) for i in range(3)])


def ranked_probability_score(y_true, probs, classes=[0, 1, 2]):
    n_classes  = len(classes)
    rps_scores = []
    for i in range(len(y_true)):
        true_one_hot = np.zeros(n_classes)
        true_one_hot[y_true.iloc[i]] = 1
        cum_pred = np.cumsum(probs[i])
        cum_true = np.cumsum(true_one_hot)
        rps_scores.append(np.sum((cum_pred - cum_true) ** 2) / (n_classes - 1))
    return np.mean(rps_scores)


def compute_ece(y_true, probs, classes=[0, 1, 2], n_bins=10):
    y_bin         = label_binarize(y_true, classes=classes)
    ece_per_class = []
    for i in range(len(classes)):
        prob_i = probs[:, i]
        true_i = y_bin[:, i]
        bins   = np.linspace(0, 1, n_bins + 1)
        ece    = 0.0
        n      = len(prob_i)
        for b in range(n_bins):
            mask = (prob_i >= bins[b]) & (prob_i < bins[b + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(true_i[mask].mean() - prob_i[mask].mean())
        ece_per_class.append(ece)
    return np.mean(ece_per_class)


print("\n===== CALIBRATION COMPARISON =====")
print(f"{'Model':25s} | {'Log Loss':>10} | {'Brier':>7} | {'ECE':>6} | {'RPS':>7}")
print("-" * 65)

for name, probs in [
    ("Uncalibrated RF",     y_prob_uncal),
    ("Platt Scaling",       y_prob_platt),
    ("Isotonic Regression", y_prob_isotonic),
]:
    ll  = log_loss(y_test, probs)
    bs  = multiclass_brier(y_test, probs)
    ece = compute_ece(y_test, probs)
    rps = ranked_probability_score(y_test, probs)
    print(f"{name:25s} | {ll:>10.4f} | {bs:>7.4f} | {ece:>6.4f} | {rps:>7.4f}")

bookie_probs      = X_test[["p_home", "p_draw", "p_away"]].values
bookie_probs_norm = bookie_probs / bookie_probs.sum(axis=1, keepdims=True)
bookie_ll         = log_loss(y_test, bookie_probs_norm)
bookie_bs         = multiclass_brier(y_test, bookie_probs_norm)
bookie_ece        = compute_ece(y_test, bookie_probs_norm)
bookie_rps        = ranked_probability_score(y_test, bookie_probs_norm)
print(f"{'Bookmaker':25s} | {bookie_ll:>10.4f} | {bookie_bs:>7.4f} | {bookie_ece:>6.4f} | {bookie_rps:>7.4f}")

# ---------------------------------------------------------------------------
# Reliability diagrams
# ---------------------------------------------------------------------------

class_names = ["Home Win", "Draw", "Away Win"]
fig, axes   = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    for name, probs, color in [
        ("Uncalibrated",  y_prob_uncal,      "steelblue"),
        ("Platt",         y_prob_platt,      "darkorange"),
        ("Isotonic",      y_prob_isotonic,   "green"),
        ("Bookmaker",     bookie_probs_norm, "red"),
    ]:
        frac_pos, mean_pred = calibration_curve(
            (y_test == i).astype(int), probs[:, i], n_bins=8
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=name, color=color)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title(f"Calibration — {class_names[i]}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Random Forest Calibration", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()