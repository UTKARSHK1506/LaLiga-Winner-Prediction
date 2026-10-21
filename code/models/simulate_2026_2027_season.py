import os
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax
from scipy.optimize import minimize
import random

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORICAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data", "la_liga_data.csv")
RAW_2627_PATH = os.path.join(BASE_DIR, "data", "raw_data", "SP1 (1).csv")
OUTPUT_FULL_380_PATH = os.path.join(BASE_DIR, "data", "processed_data", "la_liga_2627_full_380_predictions.csv")
OUTPUT_TABLE_PATH = os.path.join(BASE_DIR, "data", "processed_data", "la_liga_2627_projected_table.csv")

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

REVERSE_TEAM_MAP = {v: k for k, v in TEAM_MAP_2627.items()}

FEATURES = [
    "p_home", "p_draw", "p_away",
    "elo_diff",
    "form_diff", "goals_form_diff", "defensive_form_diff",
    "deep_diff", "xpts_diff", "ppda_diff", "sot_diff",
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
]

def elo_to_prob(elo_home, elo_away, home_advantage=65.0):
    diff = elo_home + home_advantage - elo_away
    p_win_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    p_draw = 0.26 * (1.0 - abs(p_win_home - 0.5) * 1.2)
    p_home = p_win_home * (1.0 - p_draw)
    p_away = (1.0 - p_win_home) * (1.0 - p_draw)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total

def train_ensemble_model(hist_df):
    mapping = {"H": 0, "D": 1, "A": 2}
    hist_df["target"] = hist_df["FTR"].map(mapping)
    train_df = hist_df[FEATURES + ["target"]].dropna()
    X = train_df[FEATURES]
    y = train_df["target"]
    
    print(f"Training Ensemble Model on {len(X)} historical matches...")
    
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
    xgb.fit(X, y)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X, y)
    
    lr = CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=1000), cv=5)
    lr.fit(X, y)
    
    def predict_ensemble(X_test):
        p_xgb = xgb.predict_proba(X_test)
        p_rf = rf.predict_proba(X_test)
        p_lr = lr.predict_proba(X_test)
        # Weighted blend: XGB 50%, LogReg 30%, RF 20%
        p_blend = 0.50 * p_xgb + 0.30 * p_lr + 0.20 * p_rf
        
        # Temperature Scaling T=0.90
        log_p = np.log(np.clip(p_blend, 1e-10, 1.0))
        p_temp = softmax(log_p / 0.90, axis=1)
        return p_temp
        
    return predict_ensemble

def generate_full_380_fixtures(played_df, teams_20):
    played_pairs = set()
    for _, r in played_df.iterrows():
        played_pairs.add((r["homeTeam"], r["awayTeam"]))
        
    all_pairs = []
    for h in teams_20:
        for a in teams_20:
            if h != a:
                all_pairs.append((h, a))
                
    unplayed_pairs = [pair for pair in all_pairs if pair not in played_pairs]
    
    start_date = played_df["matchDate"].max() + pd.Timedelta(days=4)
    future_rows = []
    current_date = start_date
    
    for idx, (h, a) in enumerate(unplayed_pairs):
        if idx % 10 == 0 and idx > 0:
            current_date += pd.Timedelta(days=7)
            
        future_rows.append({
            "matchDate": current_date + pd.Timedelta(days=random.choice([0, 1, 2])),
            "homeTeam": h,
            "awayTeam": a,
            "HomeTeam": REVERSE_TEAM_MAP.get(h, h.title()),
            "AwayTeam": REVERSE_TEAM_MAP.get(a, a.title()),
            "is_future": True,
            "FTR": None,
            "FTHG": None,
            "FTAG": None,
        })
        
    future_df = pd.DataFrame(future_rows)
    
    played_df["is_future"] = False
    full_df = pd.concat([played_df, future_df], ignore_index=True)
    full_df = full_df.sort_values("matchDate").reset_index(drop=True)
    return full_df

def run_season_simulation():
    print("Loading datasets...")
    hist_df = pd.read_csv(HISTORICAL_DATA_PATH)
    hist_df["matchDate"] = pd.to_datetime(hist_df["matchDate"])
    
    raw_2627 = pd.read_csv(RAW_2627_PATH)
    raw_2627["matchDate"] = pd.to_datetime(raw_2627["Date"], format="%d/%m/%Y")
    raw_2627["homeTeam"] = raw_2627["HomeTeam"].replace(TEAM_MAP_2627).str.lower().str.strip()
    raw_2627["awayTeam"] = raw_2627["AwayTeam"].replace(TEAM_MAP_2627).str.lower().str.strip()
    
    odds_h = raw_2627["B365H"].fillna(raw_2627["AvgH"])
    odds_d = raw_2627["B365D"].fillna(raw_2627["AvgD"])
    odds_a = raw_2627["B365A"].fillna(raw_2627["AvgA"])
    tot = (1/odds_h) + (1/odds_d) + (1/odds_a)
    raw_2627["p_home"] = (1/odds_h) / tot
    raw_2627["p_draw"] = (1/odds_d) / tot
    raw_2627["p_away"] = (1/odds_a) / tot
    raw_2627["home_xg"] = raw_2627["HxG"].fillna(raw_2627["FTHG"])
    raw_2627["away_xg"] = raw_2627["AxG"].fillna(raw_2627["FTAG"])
    raw_2627["home_sot"] = raw_2627["HST"].fillna(0)
    raw_2627["away_sot"] = raw_2627["AST"].fillna(0)
    
    teams_20 = list(set(raw_2627["homeTeam"]).union(set(raw_2627["awayTeam"])))
    print(f"La Liga 2026/27 teams identified ({len(teams_20)} teams): {', '.join(sorted(teams_20))}")
    
    predict_ensemble = train_ensemble_model(hist_df)
    
    full_schedule = generate_full_380_fixtures(raw_2627, teams_20)
    print(f"Generated complete 380-match schedule ({len(raw_2627)} played + {len(full_schedule)-len(raw_2627)} future fixtures).")
    
    # Initialize team state tracker
    team_elo = {}
    team_stats = {t: {"pts_hist": [], "gf_hist": [], "ga_hist": [], "xg_hist": [], "xga_hist": [], "sot_hist": []} for t in teams_20}
    
    # Initialize ELO from historical data
    for t in teams_20:
        h = hist_df[hist_df["homeTeam"] == t]
        a = hist_df[hist_df["awayTeam"] == t]
        if not h.empty:
            team_elo[t] = float(h["elo_home_pre"].values[-1])
        elif not a.empty:
            team_elo[t] = float(a["elo_away_pre"].values[-1])
        else:
            team_elo[t] = 1500.0
            
    predictions = []
    
    # Sequential prediction & simulation
    for idx in range(len(full_schedule)):
        row = full_schedule.iloc[idx]
        h_team = row["homeTeam"]
        a_team = row["awayTeam"]
        is_fut = row["is_future"]
        
        # Calculate features dynamically
        elo_h = team_elo[h_team]
        elo_a = team_elo[a_team]
        elo_diff = elo_h - elo_a
        
        if is_fut:
            p_h_elo, p_d_elo, p_a_elo = elo_to_prob(elo_h, elo_a)
            p_h, p_d, p_a = p_h_elo, p_d_elo, p_a_elo
        else:
            p_h, p_d, p_a = row["p_home"], row["p_draw"], row["p_away"]
            
        def get_rolling(team, key, default, n=5):
            vals = team_stats[team][key][-n:]
            if len(vals) == 0:
                return default
            return sum(vals) / len(vals)
            
        h_pts = get_rolling(h_team, "pts_hist", 1.4) * 5
        a_pts = get_rolling(a_team, "pts_hist", 1.3) * 5
        form_diff = h_pts - a_pts
        
        h_gf = get_rolling(h_team, "gf_hist", 1.3)
        a_gf = get_rolling(a_team, "gf_hist", 1.2)
        goals_form_diff = h_gf - a_gf
        
        h_ga = get_rolling(h_team, "ga_hist", 1.2)
        a_ga = get_rolling(a_team, "ga_hist", 1.3)
        defensive_form_diff = a_ga - h_ga
        
        h_xg = get_rolling(h_team, "xg_hist", 1.35)
        a_xg = get_rolling(a_team, "xg_hist", 1.25)
        h_xga = get_rolling(h_team, "xga_hist", 1.25)
        a_xga = get_rolling(a_team, "xga_hist", 1.35)
        
        h_sot = get_rolling(h_team, "sot_hist", 4.2)
        a_sot = get_rolling(a_team, "sot_hist", 3.8)
        sot_diff = h_sot - a_sot
        
        deep_diff = (h_xg - a_xg) * 3.5
        xpts_diff = (p_h - p_a) * 2.5
        ppda_diff = (a_xga - h_xga) * 2.0
        
        feat_vector = pd.DataFrame([{
            "p_home": p_h, "p_draw": p_d, "p_away": p_a,
            "elo_diff": elo_diff,
            "form_diff": form_diff, "goals_form_diff": goals_form_diff, "defensive_form_diff": defensive_form_diff,
            "deep_diff": deep_diff, "xpts_diff": xpts_diff, "ppda_diff": ppda_diff, "sot_diff": sot_diff,
            "home_xg_avg": h_xg, "away_xg_avg": a_xg,
            "home_xga_avg": h_xga, "away_xga_avg": a_xga,
        }])
        
        probs = predict_ensemble(feat_vector)[0]
        p_home_ens, p_draw_ens, p_away_ens = probs
        
        outcomes = ["H", "D", "A"]
        pred_class_idx = np.argmax(probs)
        pred_outcome = outcomes[pred_class_idx]
        
        if not is_fut:
            act_outcome = row["FTR"]
            act_hg = int(row["FTHG"])
            act_ag = int(row["FTAG"])
            actual_str = f"{act_hg}-{act_ag}"
        else:
            act_outcome = pred_outcome
            if pred_outcome == "H":
                act_hg, act_ag = (2, 1) if p_home_ens < 0.6 else (3, 0)
            elif pred_outcome == "A":
                act_hg, act_ag = (1, 2) if p_away_ens < 0.6 else (0, 3)
            else:
                act_hg, act_ag = (1, 1)
            actual_str = f"{act_hg}-{act_ag} (Simulated)"
            
        # Update team state
        h_p = 3 if act_outcome == "H" else (1 if act_outcome == "D" else 0)
        a_p = 3 if act_outcome == "A" else (1 if act_outcome == "D" else 0)
        
        team_stats[h_team]["pts_hist"].append(h_p)
        team_stats[a_team]["pts_hist"].append(a_p)
        team_stats[h_team]["gf_hist"].append(act_hg)
        team_stats[a_team]["gf_hist"].append(act_ag)
        team_stats[h_team]["ga_hist"].append(act_ag)
        team_stats[a_team]["ga_hist"].append(act_hg)
        team_stats[h_team]["xg_hist"].append(row["home_xg"] if not is_fut else (1.5 if act_outcome=="H" else 1.0))
        team_stats[a_team]["xg_hist"].append(row["away_xg"] if not is_fut else (1.5 if act_outcome=="A" else 1.0))
        team_stats[h_team]["sot_hist"].append(row["home_sot"] if not is_fut else 5)
        team_stats[a_team]["sot_hist"].append(row["away_sot"] if not is_fut else 4)
        
        # ELO update
        K = 20.0
        exp_h = 1.0 / (1.0 + 10.0 ** (-(elo_h + 65.0 - elo_a) / 400.0))
        actual_h_score = 1.0 if act_outcome == "H" else (0.5 if act_outcome == "D" else 0.0)
        elo_change = K * (actual_h_score - exp_h)
        team_elo[h_team] += elo_change
        team_elo[a_team] -= elo_change
        
        full_names = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
        
        predictions.append({
            "Match_ID": idx + 1,
            "Matchday": (idx // 10) + 1,
            "Date": row["matchDate"].strftime("%Y-%m-%d"),
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "Is_Played": "YES" if not is_fut else "NO",
            "Actual_Result": row["FTR"] if not is_fut else "-",
            "Scoreline": actual_str,
            "Predicted_Result": pred_outcome,
            "Predicted_Outcome": full_names[pred_outcome],
            "Home_Win_Prob": f"{p_home_ens*100:.1f}%",
            "Draw_Prob": f"{p_draw_ens*100:.1f}%",
            "Away_Win_Prob": f"{p_away_ens*100:.1f}%",
            "Home_Prob_Val": p_home_ens,
            "Draw_Prob_Val": p_draw_ens,
            "Away_Prob_Val": p_away_ens,
        })
        
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_FULL_380_PATH, index=False)
    print(f"Saved all 380 match predictions to: {OUTPUT_FULL_380_PATH}")
    
    # -----------------------------------------------------------------------
    # Monte Carlo Season Simulation (1,000 Iterations)
    # -----------------------------------------------------------------------
    print("\nRunning 1,000 Monte Carlo Season Simulations to compute Final Standings & Trophy/Relegation Odds...")
    
    N_SIMS = 1000
    sim_results = {t: {"pts": [], "wins": [], "draws": [], "losses": [], "gf": [], "ga": [], "rank": []} for t in teams_20}
    
    for sim in range(N_SIMS):
        table = {t: {"pts": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0} for t in teams_20}
        
        for p in predictions:
            h = TEAM_MAP_2627.get(p["HomeTeam"], p["HomeTeam"].lower())
            a = TEAM_MAP_2627.get(p["AwayTeam"], p["AwayTeam"].lower())
            
            if p["Is_Played"] == "YES":
                res = p["Actual_Result"]
                parts = p["Scoreline"].split("-")
                hg, ag = int(parts[0]), int(parts[1])
            else:
                # Stochastic Monte Carlo draw from predicted probabilities
                rnd = random.random()
                ph, pd_val = p["Home_Prob_Val"], p["Draw_Prob_Val"]
                if rnd < ph:
                    res = "H"
                    hg, ag = 2, 1
                elif rnd < ph + pd_val:
                    res = "D"
                    hg, ag = 1, 1
                else:
                    res = "A"
                    hg, ag = 1, 2
                    
            if res == "H":
                table[h]["pts"] += 3; table[h]["wins"] += 1; table[a]["losses"] += 1
            elif res == "A":
                table[a]["pts"] += 3; table[a]["wins"] += 1; table[h]["losses"] += 1
            else:
                table[h]["pts"] += 1; table[h]["draws"] += 1; table[a]["pts"] += 1; table[a]["draws"] += 1
                
            table[h]["gf"] += hg; table[h]["ga"] += ag
            table[a]["gf"] += ag; table[a]["ga"] += hg
            
        # Rank teams in this simulation
        sorted_teams = sorted(teams_20, key=lambda t: (table[t]["pts"], table[t]["gf"] - table[t]["ga"], table[t]["gf"]), reverse=True)
        
        for rank, t in enumerate(sorted_teams, 1):
            sim_results[t]["pts"].append(table[t]["pts"])
            sim_results[t]["wins"].append(table[t]["wins"])
            sim_results[t]["draws"].append(table[t]["draws"])
            sim_results[t]["losses"].append(table[t]["losses"])
            sim_results[t]["gf"].append(table[t]["gf"])
            sim_results[t]["ga"].append(table[t]["ga"])
            sim_results[t]["rank"].append(rank)
            
    # Aggregate Monte Carlo Summary
    summary = []
    for t in teams_20:
        ranks = sim_results[t]["rank"]
        title_pct = sum(1 for r in ranks if r == 1) / N_SIMS * 100.0
        top4_pct  = sum(1 for r in ranks if r <= 4) / N_SIMS * 100.0
        releg_pct = sum(1 for r in ranks if r >= 18) / N_SIMS * 100.0
        
        display_name = REVERSE_TEAM_MAP.get(t, t.title())
        
        summary.append({
            "Team": display_name,
            "Avg_Points": np.mean(sim_results[t]["pts"]),
            "Avg_Wins": np.mean(sim_results[t]["wins"]),
            "Avg_Draws": np.mean(sim_results[t]["draws"]),
            "Avg_Losses": np.mean(sim_results[t]["losses"]),
            "Avg_GF": np.mean(sim_results[t]["gf"]),
            "Avg_GA": np.mean(sim_results[t]["ga"]),
            "Goal_Diff": np.mean(sim_results[t]["gf"]) - np.mean(sim_results[t]["ga"]),
            "Title_Winner_Prob": title_pct,
            "Top4_UCL_Prob": top4_pct,
            "Relegation_Prob": releg_pct,
        })
        
    summary_df = pd.DataFrame(summary).sort_values("Avg_Points", ascending=False).reset_index(drop=True)
    summary_df.index += 1
    summary_df.to_csv(OUTPUT_TABLE_PATH, index_label="Pos")
    
    print("\n=======================================================")
    print("PROJECTED FINAL 2026/27 LA LIGA LEAGUE TABLE (38 MATCHDAYS)")
    print("=======================================================")
    print(summary_df[["Team", "Avg_Points", "Avg_Wins", "Avg_Draws", "Avg_Losses", "Goal_Diff", "Title_Winner_Prob", "Top4_UCL_Prob", "Relegation_Prob"]].to_string())

if __name__ == "__main__":
    run_season_simulation()
