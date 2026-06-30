import os
import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data")
INPUT_PATH  = os.path.join(DATA_DIR, "la_liga_data.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "la_liga_data.csv")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Result distribution
# ---------------------------------------------------------------------------

print("Result distribution:")
print(df["FTR"].value_counts())
print(df["FTR"].value_counts(normalize=True))
print()

sns.countplot(x="FTR", data=df)
plt.title("Full time result distribution")
plt.show()

sns.boxplot(x="FTR", y="elo_diff", data=df)
plt.title("ELO difference vs match result")
plt.show()

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

strength_features    = ["elo_diff", "overall_diff", "attack_diff", "midfield_diff", "defence_diff"]
form_features        = ["form_diff", "goals_form_diff", "defensive_form_diff"]
performance_features = ["ppda_diff", "deep_diff", "xpts_diff"]
context_features     = ["rest_diff", "venue_bias_diff"]
betting_features     = ["p_home", "p_draw", "p_away"]

# ---------------------------------------------------------------------------
# Mean by result for each feature group
# ---------------------------------------------------------------------------

print("--- Strength features ---")
for col in strength_features:
    print(col)
    print(df.groupby("FTR")[col].mean())
    print()

print("--- Form features ---")
for col in form_features:
    print(col)
    print(df.groupby("FTR")[col].mean())
    print()

print("--- Performance features ---")
for col in performance_features:
    print(col)
    print(df.groupby("FTR")[col].mean())
    print()

print("--- Context features ---")
for col in context_features:
    print(col)
    print(df.groupby("FTR")[col].mean())
    print()

# ---------------------------------------------------------------------------
# KDE plots — key features
# ---------------------------------------------------------------------------

important_features = ["elo_diff", "form_diff", "goals_form_diff", "deep_diff", "xpts_diff"]

for col in important_features:
    sns.kdeplot(data=df, x=col, hue="FTR", fill=True)
    plt.title(col)
    plt.show()

sns.scatterplot(data=df, x="elo_diff", y="form_diff", hue="FTR", alpha=0.5)
plt.title("ELO diff vs form diff")
plt.show()

# ---------------------------------------------------------------------------
# Odds baseline accuracy + confidence analysis
# ---------------------------------------------------------------------------

print("Mean odds probabilities by result:")
print(df.groupby("FTR")[["p_home", "p_draw", "p_away"]].mean())
print()

print(f"Odds baseline accuracy: {(df['predicted'] == df['FTR']).mean():.4f}")

print("\nAccuracy by confidence quintile:")
print(
    df.groupby(pd.qcut(df["confidence"], 5))["predicted"]
    .apply(lambda x: (x == df.loc[x.index, "FTR"]).mean())
)
print()

sns.scatterplot(data=df, x="elo_diff", y="p_home", alpha=0.5)
plt.title("ELO diff vs p_home")
plt.show()

# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

selected_cols = [
    "elo_diff", "overall_diff", "attack_diff", "midfield_diff", "defence_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "ppda_diff", "xpts_diff",
    "venue_bias_diff", "p_home", "p_draw",
]

corr = df[selected_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.show()

# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

X_vif = df[selected_cols].dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]
print("VIF:")
print(vif_data.sort_values("VIF", ascending=False))
print()

# ---------------------------------------------------------------------------
# Cramér's V
# ---------------------------------------------------------------------------

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

print(f"Cramér's V (FTR vs Season): {cramers_v(df['FTR'], df['Season']):.4f}")
print()

# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------

print("ANOVA p-values:")
for col in ["elo_diff", "form_diff", "deep_diff"]:
    groups = [group[col].dropna() for _, group in df.groupby("FTR")]
    stat, p = f_oneway(*groups)
    print(f"  {col}: p = {p:.6f}")
print()

# ---------------------------------------------------------------------------
# Save (overwrites with same data — confirms pipeline ran cleanly)
# ---------------------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {df.shape}")
