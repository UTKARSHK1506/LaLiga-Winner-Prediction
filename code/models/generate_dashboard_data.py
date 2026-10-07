import os
import json
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORICAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data", "la_liga_data.csv")
RAW_70_PATH = os.path.join(BASE_DIR, "data", "raw_data", "SP1 (1).csv")
DASHBOARD_DATA_PATH = os.path.join(BASE_DIR, "dashboard", "data.json")

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
    "La Coruna": "la coruna",
    "Levante": "levante",
    "Malaga": "malaga",
    "Osasuna": "osasuna",
    "Real Madrid": "real madrid",
    "Santander": "sp gijon",
    "Sevilla": "sevilla",
    "Sociedad": "sociedad",
    "Valencia": "valencia",
    "Vallecano": "vallecano",
    "Villarreal": "villarreal",
}

REVERSE_TEAM_MAP = {v: k for k, v in TEAM_MAP_2627.items()}

FEATURES = [
    "p_home", "p_draw", "p_away",
    "elo_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "xpts_diff", "ppda_diff", "sot_diff",
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
]

def load_and_preprocess_70(filepath, historical_df):
    df_raw = pd.read_csv(filepath)
    df = df_raw.copy()
    
    df["matchDate"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values("matchDate").reset_index(drop=True)
    
    df["homeTeam"] = df["HomeTeam"].replace(TEAM_MAP_2627).str.lower().str.strip()
    df["awayTeam"] = df["AwayTeam"].replace(TEAM_MAP_2627).str.lower().str.strip()
    
    odds_h = df["B365H"].fillna(df["AvgH"])
    odds_d = df["B365D"].fillna(df["AvgD"])
    odds_a = df["B365A"].fillna(df["AvgA"])
    
    tot = (1/odds_h) + (1/odds_d) + (1/odds_a)
    df["p_home"] = (1/odds_h) / tot
    df["p_draw"] = (1/odds_d) / tot
    df["p_away"] = (1/odds_a) / tot
    
    df["home_sot"] = df["HST"].fillna(0)
    df["away_sot"] = df["AST"].fillna(0)
    df["home_xg"] = df["HxG"].fillna(df["FTHG"])
    df["away_xg"] = df["AxG"].fillna(df["FTAG"])
    
    latest_elo = {}
    for team in set(df["homeTeam"]).union(set(df["awayTeam"])):
        h_matches = historical_df[historical_df["homeTeam"] == team]
        a_matches = historical_df[historical_df["awayTeam"] == team]
        if not h_matches.empty and not a_matches.empty:
            l_h = h_matches["matchDate"].max()
            l_a = a_matches["matchDate"].max()
            if l_h >= l_a:
                latest_elo[team] = h_matches[h_matches["matchDate"] == l_h]["elo_home_pre"].values[0]
            else:
                latest_elo[team] = a_matches[a_matches["matchDate"] == l_a]["elo_away_pre"].values[0]
        elif not h_matches.empty:
            latest_elo[team] = h_matches["elo_home_pre"].values[-1]
        elif not a_matches.empty:
            latest_elo[team] = a_matches["elo_away_pre"].values[-1]
        else:
            latest_elo[team] = 1500.0
            
    df["elo_home"] = df["homeTeam"].map(latest_elo).fillna(1500.0)
    df["elo_away"] = df["awayTeam"].map(latest_elo).fillna(1500.0)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    
    teams = list(set(df["homeTeam"]).union(set(df["awayTeam"])))
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
        
        t_df["pts_last5"] = t_df["pts_last5"].fillna(6.5)
        t_df["gf_last5"]  = t_df["gf_last5"].fillna(1.3)
        t_df["ga_last5"]  = t_df["ga_last5"].fillna(1.2)
        t_df["xg_last5"]  = t_df["xg_last5"].fillna(1.35)
        t_df["xga_last5"] = t_df["xga_last5"].fillna(1.25)
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
    
    df["deep_diff"] = (df["home_xg_avg"] - df["away_xg_avg"]) * 3.5
    df["xpts_diff"] = (df["p_home"] - df["p_away"]) * 2.5
    df["ppda_diff"] = (df["away_xga_avg"] - df["home_xga_avg"]) * 2.0
    
    mapping = {"H": 0, "D": 1, "A": 2}
    df["target"] = df["FTR"].map(mapping)
    return df

def generate_dashboard_dataset():
    os.makedirs(os.path.join(BASE_DIR, "dashboard"), exist_ok=True)
    hist_df = pd.read_csv(HISTORICAL_DATA_PATH)
    hist_df["matchDate"] = pd.to_datetime(hist_df["matchDate"])
    
    mapping = {"H": 0, "D": 1, "A": 2}
    hist_df["target"] = hist_df["FTR"].map(mapping)
    train_df = hist_df[FEATURES + ["target"]].dropna()
    
    X_train = train_df[FEATURES]
    y_train = train_df["target"]
    
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=500,
        max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=8, reg_alpha=0.8, reg_lambda=1.7, eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train)
    
    df_70 = load_and_preprocess_70(RAW_70_PATH, hist_df)
    X_70 = df_70[FEATURES]
    y_70 = df_70["target"]
    
    probs_xgb = xgb.predict_proba(X_70)
    preds_xgb = xgb.predict(X_70)
    
    acc_xgb = accuracy_score(y_70, preds_xgb)
    
    teams = list(set(df_70["homeTeam"]).union(set(df_70["awayTeam"])))
    actual_standings = {REVERSE_TEAM_MAP.get(t, t.title()): {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "PTS": 0} for t in teams}
    
    inv_map = {0: "H", 1: "D", 2: "A"}
    full_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    
    match_list = []
    
    for i in range(len(df_70)):
        row = df_70.iloc[i]
        ph, pd_val, pa = probs_xgb[i]
        pred_code = inv_map[preds_xgb[i]]
        act_code = row["FTR"]
        
        h_name = REVERSE_TEAM_MAP.get(row["homeTeam"], row["homeTeam"].title())
        a_name = REVERSE_TEAM_MAP.get(row["awayTeam"], row["awayTeam"].title())
        
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        is_correct = (pred_code == act_code)
        
        actual_standings[h_name]["P"] += 1; actual_standings[a_name]["P"] += 1
        actual_standings[h_name]["GF"] += hg; actual_standings[h_name]["GA"] += ag
        actual_standings[a_name]["GF"] += ag; actual_standings[a_name]["GA"] += hg
        
        if act_code == "H":
            actual_standings[h_name]["PTS"] += 3; actual_standings[h_name]["W"] += 1; actual_standings[a_name]["L"] += 1
        elif act_code == "A":
            actual_standings[a_name]["PTS"] += 3; actual_standings[a_name]["W"] += 1; actual_standings[h_name]["L"] += 1
        else:
            actual_standings[h_name]["PTS"] += 1; actual_standings[h_name]["D"] += 1
            actual_standings[a_name]["PTS"] += 1; actual_standings[a_name]["D"] += 1
            
        match_list.append({
            "id": i + 1,
            "matchday": (i // 10) + 1,
            "date": row["matchDate"].strftime("%b %d, %Y"),
            "home": h_name,
            "away": a_name,
            "played": True,
            "actual_result": act_code,
            "score": f"{hg}-{ag}",
            "pred_code": pred_code,
            "pred_text": full_map[pred_code],
            "p_home": float(round(ph * 100, 1)),
            "p_draw": float(round(pd_val * 100, 1)),
            "p_away": float(round(pa * 100, 1)),
            "correct": is_correct,
        })
        
    def format_table(t_dict):
        res = []
        for team, stats in t_dict.items():
            gd = stats["GF"] - stats["GA"]
            res.append({
                "team": team,
                "P": stats["P"], "W": stats["W"], "D": stats["D"], "L": stats["L"],
                "GF": stats["GF"], "GA": stats["GA"], "GD": gd, "PTS": stats["PTS"]
            })
        df_t = pd.DataFrame(res).sort_values(["PTS", "GD", "W"], ascending=False).reset_index(drop=True)
        df_t["Pos"] = df_t.index + 1
        return df_t.to_dict(orient="records")
        
    final_json = {
        "summary": {
            "total_matches": len(df_70),
            "played_matches": len(df_70),
            "played_accuracy": float(round(acc_xgb * 100, 2)),
            "model_name": "Standalone XGBoost Classifier"
        },
        "matches": match_list,
        "actual_standings": format_table(actual_standings),
    }
    
    with open(DASHBOARD_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)
        
    print(f"Successfully generated clean rounded dashboard dataset JSON at: {DASHBOARD_DATA_PATH}")

if __name__ == "__main__":
    generate_dashboard_dataset()
