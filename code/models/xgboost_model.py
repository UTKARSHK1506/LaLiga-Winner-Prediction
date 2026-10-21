import os
import pandas as pd
import numpy as np
import shap
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, classification_report, log_loss
from scipy.optimize import minimize
from scipy.special import softmax

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed_data", "la_liga_data.csv")

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)
df["Season"] = df["Season"].astype(str).str[:4].astype(int)

df["odds_diff"]             = df["p_home"] - df["p_away"]
df["xg_strength_gap"]       = df["home_xg_avg"] - df["away_xg_avg"]
df["dominance"]             = df["deep_diff"] + df["ppda_diff"]
df["odds_xpts_interaction"] = df["p_home"] * df["xpts_diff"]
df["pressure_interaction"]  = df["ppda_diff"] * df["deep_diff"]
df["elo_xg_interaction"]    = df["elo_diff"] * df["xg_vs_def_diff"]

features = [
    "p_home", "p_draw", "p_away",
    "elo_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "xpts_diff", "ppda_diff", "sot_diff",
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
]

mapping = {"H": 0, "D": 1, "A": 2}
df["target"] = df["FTR"].map(mapping)
class_weights = {0: 1.0, 1: 0.8, 2: 1.0}

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

model_df = df[features + ["target", "Season"]].dropna()

TRAIN_SEASONS_MAX = 2023
TEST_SEASON       = 2024

train = model_df[model_df["Season"] <= TRAIN_SEASONS_MAX]
test  = model_df[model_df["Season"] == TEST_SEASON]

X_train = train[features]
y_train = train["target"]
X_test  = test[features]
y_test  = test["target"]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_xgb_model():
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=8,
        reg_alpha=0.8,
        reg_lambda=1.7,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

xgb = get_xgb_model()
xgb.fit(X_train, y_train)
y_prob = xgb.predict_proba(X_test)
y_pred = xgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

feat_imp = pd.DataFrame({
    "feature":    features,
    "importance": xgb.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(feat_imp)

# ---------------------------------------------------------------------------
# Draw analysis
# ---------------------------------------------------------------------------

test_df = X_test.copy()
test_df["true"] = y_test
test_df["pred"] = y_pred
draw_cases = test_df[test_df["pred"] == 1]
print(draw_cases.head())

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

try:
    explainer   = shap.Explainer(xgb, X_train[features])
    shap_values = explainer(X_test[features])
except Exception as e:
    print(f"SHAP calculation skipped: {e}")

# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

CLASS_NAMES  = ["Home Win", "Draw", "Away Win"]
probs        = xgb.predict_proba(X_test)
y_true       = np.array(y_test)
probs_book   = X_test[["p_home", "p_draw", "p_away"]].values

print("Probability shape:", probs.shape)
print("Sample probs (first 3 rows):")
print(np.round(probs[:3], 3))
print("Row sums:", probs[:5].sum(axis=1))


def compute_ece(y_true, y_prob, n_bins=10):
    ece_per_class = []
    for c in range(y_prob.shape[1]):
        y_bin  = (y_true == c).astype(int)
        prob_c = y_prob[:, c]
        bins   = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (prob_c >= bins[i]) & (prob_c < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc  = y_bin[mask].mean()
            conf = prob_c[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(acc - conf)
        ece_per_class.append(ece)
    return np.mean(ece_per_class), ece_per_class


def compute_mce(y_true, y_prob, n_bins=10):
    mce_per_class = []
    for c in range(y_prob.shape[1]):
        y_bin   = (y_true == c).astype(int)
        prob_c  = y_prob[:, c]
        bins    = np.linspace(0, 1, n_bins + 1)
        max_err = 0.0
        for i in range(n_bins):
            mask = (prob_c >= bins[i]) & (prob_c < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc     = y_bin[mask].mean()
            conf    = prob_c[mask].mean()
            max_err = max(max_err, abs(acc - conf))
        mce_per_class.append(max_err)
    return np.mean(mce_per_class), mce_per_class


def compute_brier(y_true, y_prob):
    scores = []
    for c in range(y_prob.shape[1]):
        y_bin = (y_true == c).astype(float)
        scores.append(np.mean((y_prob[:, c] - y_bin) ** 2))
    return np.mean(scores), scores


def compute_rps(y_true, y_prob):
    n_classes = y_prob.shape[1]
    rps_total = 0.0
    for i in range(len(y_true)):
        outcome = np.zeros(n_classes)
        outcome[y_true[i]] = 1
        rps_total += np.sum((np.cumsum(y_prob[i]) - np.cumsum(outcome)) ** 2) / (n_classes - 1)
    return rps_total / len(y_true)


def compute_sharpness(y_prob):
    return y_prob.max(axis=1).mean()


def compute_kl(p, q, eps=1e-10):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.mean(np.sum(p * np.log(p / q), axis=1))


ece_overall, ece_per_class   = compute_ece(y_true, probs)
mce_overall, mce_per_class   = compute_mce(y_true, probs)
brier_overall, brier_per_cls = compute_brier(y_true, probs)
rps        = compute_rps(y_true, probs)
sharpness  = compute_sharpness(probs)
ll         = log_loss(y_true, probs)
kl_vs_book = compute_kl(probs, probs_book)

ece_book, _   = compute_ece(y_true, probs_book)
brier_book, _ = compute_brier(y_true, probs_book)
rps_book      = compute_rps(y_true, probs_book)
sharp_book    = compute_sharpness(probs_book)
ll_book       = log_loss(y_true, probs_book)

print("=" * 55)
print("CALIBRATION METRICS - XGBoost (Uncalibrated)")
print("=" * 55)
print(f"Log Loss:              {ll:.4f}  (Bookmaker: {ll_book:.4f})")
print(f"Brier Score (overall): {brier_overall:.4f}  (Bookmaker: {brier_book:.4f})")
print(f"RPS:                   {rps:.4f}  (Bookmaker: {rps_book:.4f})")
print(f"ECE (overall):         {ece_overall:.4f}  (Bookmaker: {ece_book:.4f})")
print(f"MCE (overall):         {mce_overall:.4f}")
print(f"Sharpness:             {sharpness:.4f}  (Bookmaker: {sharp_book:.4f})")
print(f"KL Div vs Bookmaker:   {kl_vs_book:.4f}")
print()
print("--- Per-Class Breakdown ---")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<12}  ECE={ece_per_class[i]:.4f}  MCE={mce_per_class[i]:.4f}  Brier={brier_per_cls[i]:.4f}")

# ---------------------------------------------------------------------------
# Calibration comparison
# ---------------------------------------------------------------------------

cal_platt    = CalibratedClassifierCV(get_xgb_model(), cv=5, method="sigmoid")
cal_isotonic = CalibratedClassifierCV(get_xgb_model(), cv=5, method="isotonic")
cal_platt.fit(X_train, y_train)
cal_isotonic.fit(X_train, y_train)
probs_platt    = cal_platt.predict_proba(X_test)
probs_isotonic = cal_isotonic.predict_proba(X_test)
print("Calibrators fitted.")

log_probs = np.log(np.clip(probs, 1e-10, 1))

def ece_loss(T):
    scaled = softmax(log_probs / T[0], axis=1)
    ece, _ = compute_ece(y_true, scaled)
    return ece

result     = minimize(ece_loss, x0=[1.5], bounds=[(0.1, 10.0)])
best_T     = result.x[0]
probs_temp = softmax(log_probs / best_T, axis=1)
print(f"Best temperature: {best_T:.4f}")

models_to_compare = {
    "Uncalibrated XGB":    probs,
    "Platt Scaling":       probs_platt,
    "Isotonic Regression": probs_isotonic,
    "Temperature Scaling": probs_temp,
    "Bookmaker":           probs_book,
}

print(f"{'Model':<25} | {'Log Loss':>10} | {'Brier':>7} | {'RPS':>7} | {'ECE':>7} | {'MCE':>7} | {'Sharpness':>10}")
print("-" * 85)

cal_results = {}
for name, p in models_to_compare.items():
    ll_c           = log_loss(y_true, p)
    b_c, _         = compute_brier(y_true, p)
    rps_c          = compute_rps(y_true, p)
    ece_c, ece_cls = compute_ece(y_true, p)
    mce_c, _       = compute_mce(y_true, p)
    sh_c           = compute_sharpness(p)
    cal_results[name] = {"ll": ll_c, "brier": b_c, "rps": rps_c,
                         "ece": ece_c, "mce": mce_c, "sharpness": sh_c,
                         "ece_per_class": ece_cls}
    print(f"{name:<25} | {ll_c:>10.4f} | {b_c:>7.4f} | {rps_c:>7.4f} | {ece_c:>7.4f} | {mce_c:>7.4f} | {sh_c:>10.4f}")

print()
print("--- Per-Class ECE ---")
for name, res in cal_results.items():
    epc = res["ece_per_class"]
    print(f"{name:<25}  Home={epc[0]:.4f}  Draw={epc[1]:.4f}  Away={epc[2]:.4f}")

# ---------------------------------------------------------------------------
# Reliability diagrams
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("XGBoost Calibration — Uncalibrated vs Calibrated vs Bookmaker",
             fontsize=11, fontweight="bold")

plot_models = {
    "Uncalibrated XGB":    (probs,       "#FF9800", "s-"),
    "Platt Scaling":       (probs_platt, "#2196F3", "^-"),
    "Temperature Scaling": (probs_temp,  "#9C27B0", "D-"),
    "Bookmaker":           (probs_book,  "#F44336", "o--"),
}

for idx, (ax, class_name) in enumerate(zip(axes, CLASS_NAMES)):
    y_bin = (y_true == idx).astype(int)
    for name, (p, color, marker) in plot_models.items():
        try:
            frac, mean_p = calibration_curve(y_bin, p[:, idx], n_bins=10, strategy="uniform")
            ax.plot(mean_p, frac, marker, color=color, lw=2, label=name)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ece_u = cal_results["Uncalibrated XGB"]["ece_per_class"][idx]
    ece_t = cal_results["Temperature Scaling"]["ece_per_class"][idx]
    ax.set_title(f"{class_name}\nECE: Uncal={ece_u:.3f} | Temp={ece_t:.3f}",
                 fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=9)
    ax.set_ylabel("Fraction of Positives", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "visualizations"), exist_ok=True)
plot_path = os.path.join(os.path.dirname(__file__), "..", "visualizations", "xgboost_calibration.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"Calibration plot saved to {plot_path}")

# ---------------------------------------------------------------------------
# Confidence vs accuracy
# ---------------------------------------------------------------------------

confidence = probs.max(axis=1)
y_pred_xgb = xgb.predict(X_test)
df_conf    = pd.DataFrame({"confidence": confidence, "correct": (y_pred_xgb == y_test.values)})
df_conf["quintile"] = pd.qcut(df_conf["confidence"], 5,
                               labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
acc_by_conf = df_conf.groupby("quintile")["correct"].mean()
conf_by_bin = df_conf.groupby("quintile")["confidence"].mean()

print("Accuracy by confidence quintile:")
for q in acc_by_conf.index:
    print(f"  {q}: accuracy={acc_by_conf[q]:.3f}, mean_confidence={conf_by_bin[q]:.3f}, gap={conf_by_bin[q]-acc_by_conf[q]:+.3f}")

kl_per_match = np.sum(
    np.clip(probs, 1e-10, 1) * np.log(np.clip(probs, 1e-10, 1) / np.clip(probs_book, 1e-10, 1)),
    axis=1,
)
print(f"\nMean KL Divergence (XGB vs Bookmaker): {kl_per_match.mean():.4f}")
print(f"Median KL Divergence:                  {np.median(kl_per_match):.4f}")
print(f"Max KL Divergence:                     {kl_per_match.max():.4f}")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print(classification_report(y_test, y_pred))