import os
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORICAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data", "la_liga_data.csv")
RAW_2627_PATH = os.path.join(BASE_DIR, "data", "raw_data", "SP1 (1).csv")
OUTPUT_PRED_PATH = os.path.join(BASE_DIR, "data", "processed_data", "la_liga_2627_predictions.csv")

# 1. Team name mapping to match historical team names
TEAM_MAP_2627 = {
    "Alaves": "alaves",
    "Ath Bilbao": "ath bilbao",
    "Ath Madrid": "ath madrid",
    "Barcelona": "barcelona",
    "Betis": "betis",
    "Celta": "celta",
    "Elche": "elche",
    "Espanol": "espanol",
    "Getafe": "getafe",
    "Girona": "girona",
    "La Coruna": "la coruna",
    "Levante": "levante",
    "Malaga": "malaga",
    "Mallorca": "mallorca",
    "Osasuna": "osasuna",
    "Real Madrid": "real madrid",
    "Santander": "sp gijon",
    "Sevilla": "sevilla",
    "Sociedad": "sociedad",
    "Valencia": "valencia",
    "Vallecano": "vallecano",
    "Villarreal": "villarreal",
}

# 2. Features used in XGBoost model
FEATURES = [
    "p_home", "p_draw", "p_away",
    "elo_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "xpts_diff", "ppda_diff", "sot_diff",
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
]

def load_and_preprocess_2627(filepath, historical_df):
    df_raw = pd.read_csv(filepath)
    df = df_raw.copy()
    
    # Parse date
    df["matchDate"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values("matchDate").reset_index(drop=True)
    
    df["homeTeam"] = df["HomeTeam"].replace(TEAM_MAP_2627).str.lower().str.strip()
    df["awayTeam"] = df["AwayTeam"].replace(TEAM_MAP_2627).str.lower().str.strip()
    
    # Probabilities from odds (B365 or Avg)
    odds_h = df["B365H"].fillna(df["AvgH"])
    odds_d = df["B365D"].fillna(df["AvgD"])
    odds_a = df["B365A"].fillna(df["AvgA"])
    
    p_h_raw = 1 / odds_h
    p_d_raw = 1 / odds_d
    p_a_raw = 1 / odds_a
    total = p_h_raw + p_d_raw + p_a_raw
    
    df["p_home"] = p_h_raw / total
    df["p_draw"] = p_d_raw / total
    df["p_away"] = p_a_raw / total
    
    # Shots on target
    df["home_sot"] = df["HST"].fillna(0)
    df["away_sot"] = df["AST"].fillna(0)
    
    # xG
    df["home_xg"] = df["HxG"].fillna(df["FTHG"])
    df["away_xg"] = df["AxG"].fillna(df["FTAG"])
    
    # Extract latest ELOs from historical data per team
    latest_elo = {}
    for team in set(df["homeTeam"]).union(set(df["awayTeam"])):
        home_matches = historical_df[historical_df["homeTeam"] == team]
        away_matches = historical_df[historical_df["awayTeam"] == team]
        if not home_matches.empty and not away_matches.empty:
            last_h_date = home_matches["matchDate"].max()
            last_a_date = away_matches["matchDate"].max()
            if last_h_date >= last_a_date:
                latest_elo[team] = home_matches[home_matches["matchDate"] == last_h_date]["elo_home_pre"].values[0]
            else:
                latest_elo[team] = away_matches[away_matches["matchDate"] == last_a_date]["elo_away_pre"].values[0]
        elif not home_matches.empty:
            latest_elo[team] = home_matches["elo_home_pre"].values[-1]
        elif not away_matches.empty:
            latest_elo[team] = away_matches["elo_away_pre"].values[-1]
        else:
            latest_elo[team] = 1500.0
            
    df["elo_home"] = df["homeTeam"].map(latest_elo).fillna(1500.0)
    df["elo_away"] = df["awayTeam"].map(latest_elo).fillna(1500.0)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    
    # Compute rolling form within 2026/27 season
    teams = list(set(df["homeTeam"]).union(set(df["awayTeam"])))
    
    # We construct team stats timeline
    stats = []
    for team in teams:
        t_matches = []
        for idx, row in df.iterrows():
            if row["homeTeam"] == team:
                pts = 3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0)
                t_matches.append({
                    "matchDate": row["matchDate"],
                    "match_idx": idx,
                    "is_home": True,
                    "pts": pts,
                    "gf": row["FTHG"],
                    "ga": row["FTAG"],
                    "xg": row["home_xg"],
                    "xga": row["away_xg"],
                    "sot": row["home_sot"],
                })
            elif row["awayTeam"] == team:
                pts = 3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0)
                t_matches.append({
                    "matchDate": row["matchDate"],
                    "match_idx": idx,
                    "is_home": False,
                    "pts": pts,
                    "gf": row["FTAG"],
                    "ga": row["FTHG"],
                    "xg": row["away_xg"],
                    "xga": row["home_xg"],
                    "sot": row["away_sot"],
                })
        t_df = pd.DataFrame(t_matches).sort_values("matchDate")
        t_df["pts_last5"] = t_df["pts"].shift(1).rolling(5, min_periods=1).mean() * 5
        t_df["gf_last5"]  = t_df["gf"].shift(1).rolling(5, min_periods=1).mean()
        t_df["ga_last5"]  = t_df["ga"].shift(1).rolling(5, min_periods=1).mean()
        t_df["xg_last5"]  = t_df["xg"].shift(1).rolling(5, min_periods=1).mean()
        t_df["xga_last5"] = t_df["xga"].shift(1).rolling(5, min_periods=1).mean()
        t_df["sot_last5"] = t_df["sot"].shift(1).rolling(5, min_periods=1).mean()
        
        # Default fallbacks for first few matches of the season using historical averages
        t_df["pts_last5"] = t_df["pts_last5"].fillna(HistoricalDefaults.get(team, {}).get("pts", 6.5))
        t_df["gf_last5"]  = t_df["gf_last5"].fillna(HistoricalDefaults.get(team, {}).get("gf", 1.3))
        t_df["ga_last5"]  = t_df["ga_last5"].fillna(HistoricalDefaults.get(team, {}).get("ga", 1.2))
        t_df["xg_last5"]  = t_df["xg_last5"].fillna(HistoricalDefaults.get(team, {}).get("xg", 1.35))
        t_df["xga_last5"] = t_df["xga_last5"].fillna(HistoricalDefaults.get(team, {}).get("xga", 1.25))
        t_df["sot_last5"] = t_df["sot_last5"].fillna(4.0)
        
        for _, r in t_df.iterrows():
            stats.append({
                "match_idx": r["match_idx"],
                "team": team,
                "is_home": r["is_home"],
                "pts_last5": r["pts_last5"],
                "gf_last5": r["gf_last5"],
                "ga_last5": r["ga_last5"],
                "xg_last5": r["xg_last5"],
                "xga_last5": r["xga_last5"],
                "sot_last5": r["sot_last5"],
            })
            
    stats_df = pd.DataFrame(stats)
    
    home_stats = stats_df[stats_df["is_home"]].set_index("match_idx")
    away_stats = stats_df[~stats_df["is_home"]].set_index("match_idx")
    
    df["form_diff"] = home_stats["pts_last5"] - away_stats["pts_last5"]
    df["goals_form_diff"] = home_stats["gf_last5"] - away_stats["gf_last5"]
    df["defensive_form_diff"] = away_stats["ga_last5"] - home_stats["ga_last5"]
    df["home_xg_avg"] = home_stats["xg_last5"]
    df["away_xg_avg"] = away_stats["xg_last5"]
    df["home_xga_avg"] = home_stats["xga_last5"]
    df["away_xga_avg"] = away_stats["xga_last5"]
    df["sot_diff"] = home_stats["sot_last5"] - away_stats["sot_last5"]
    
    # Fill remaining engineered metrics with historical mean values
    df["deep_diff"] = (df["home_xg_avg"] - df["away_xg_avg"]) * 3.5
    df["xpts_diff"] = (df["p_home"] - df["p_away"]) * 2.5
    df["ppda_diff"] = (df["away_xga_avg"] - df["home_xga_avg"]) * 2.0
    
    mapping = {"H": 0, "D": 1, "A": 2}
    df["target"] = df["FTR"].map(mapping)
    
    return df

HistoricalDefaults = {
    "real madrid": {"pts": 9.5, "gf": 2.2, "ga": 0.8, "xg": 2.1, "xga": 0.85},
    "barcelona": {"pts": 9.0, "gf": 2.1, "ga": 0.9, "xg": 2.0, "xga": 0.90},
    "ath madrid": {"pts": 8.0, "gf": 1.6, "ga": 0.8, "xg": 1.6, "xga": 0.85},
    "ath bilbao": {"pts": 7.0, "gf": 1.4, "ga": 1.0, "xg": 1.4, "xga": 1.0},
    "sociedad": {"pts": 7.0, "gf": 1.3, "ga": 1.0, "xg": 1.35, "xga": 1.0},
    "villarreal": {"pts": 6.8, "gf": 1.5, "ga": 1.2, "xg": 1.45, "xga": 1.15},
    "betis": {"pts": 6.5, "gf": 1.3, "ga": 1.2, "xg": 1.3, "xga": 1.2},
    "sevilla": {"pts": 6.0, "gf": 1.2, "ga": 1.3, "xg": 1.25, "xga": 1.25},
    "valencia": {"pts": 5.8, "gf": 1.1, "ga": 1.3, "xg": 1.2, "xga": 1.3},
    "celta": {"pts": 5.5, "gf": 1.2, "ga": 1.4, "xg": 1.2, "xga": 1.35},
    "osasuna": {"pts": 5.5, "gf": 1.1, "ga": 1.3, "xg": 1.15, "xga": 1.3},
    "getafe": {"pts": 5.0, "gf": 0.9, "ga": 1.2, "xg": 1.0, "xga": 1.2},
    "vallecano": {"pts": 5.2, "gf": 1.0, "ga": 1.3, "xg": 1.1, "xga": 1.3},
}

def main():
    print("Loading historical La Liga dataset (2015/16 - 2024/25)...")
    hist_df = pd.read_csv(HISTORICAL_DATA_PATH)
    hist_df["matchDate"] = pd.to_datetime(hist_df["matchDate"])
    
    mapping = {"H": 0, "D": 1, "A": 2}
    hist_df["target"] = hist_df["FTR"].map(mapping)
    
    train_df = hist_df[FEATURES + ["target"]].dropna()
    X_train = train_df[FEATURES]
    y_train = train_df["target"]
    
    print(f"Training XGBoost model on {len(X_train)} historical matches...")
    xgb = XGBClassifier(
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
    xgb.fit(X_train, y_train)
    print("XGBoost training complete.")
    
    print("\nProcessing 2026/27 season data from SP1 (1).csv...")
    df_2627 = load_and_preprocess_2627(RAW_2627_PATH, hist_df)
    
    X_2627 = df_2627[FEATURES]
    y_2627 = df_2627["target"]
    
    probs_2627 = xgb.predict_proba(X_2627)
    preds_2627 = xgb.predict(X_2627)
    
    acc = accuracy_score(y_2627, preds_2627)
    loss = log_loss(y_2627, probs_2627)
    
    print("\n=======================================================")
    print("2026/27 LA LIGA SEASON - XGBOOST PREDICTION RESULTS")
    print("=======================================================")
    print(f"Total Matches Evaluated: {len(df_2627)}")
    print(f"Overall Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
    print(f"Log Loss:               {loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_2627, preds_2627, target_names=["Home Win", "Draw", "Away Win"]))
    
    # Construct final detailed results dataframe
    results = []
    inv_map = {0: "H", 1: "D", 2: "A"}
    full_name_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    
    for i in range(len(df_2627)):
        row = df_2627.iloc[i]
        p_home, p_draw, p_away = probs_2627[i]
        pred_code = inv_map[preds_2627[i]]
        actual_code = row["FTR"]
        correct = (pred_code == actual_code)
        
        results.append({
            "Match_ID": i + 1,
            "Date": row["matchDate"].strftime("%Y-%m-%d"),
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "Actual_Score": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            "Actual_Result": actual_code,
            "Predicted_Result": pred_code,
            "Predicted_Outcome": full_name_map[pred_code],
            "Home_Win_Prob": f"{p_home*100:.1f}%",
            "Draw_Prob": f"{p_draw*100:.1f}%",
            "Away_Win_Prob": f"{p_away*100:.1f}%",
            "Is_Correct": "YES" if correct else "NO"
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_PRED_PATH, index=False)
    print(f"\nSaved all 2026/27 match predictions to: {OUTPUT_PRED_PATH}")
    
    print("\nFirst 20 Match Predictions:")
    print(res_df[["Match_ID", "Date", "HomeTeam", "AwayTeam", "Actual_Result", "Predicted_Result", "Home_Win_Prob", "Draw_Prob", "Away_Win_Prob", "Is_Correct"]].to_string(index=False))

if __name__ == "__main__":
    main()
