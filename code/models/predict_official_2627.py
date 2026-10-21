import os
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORICAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data", "la_liga_data.csv")
RAW_2627_PATH = os.path.join(BASE_DIR, "data", "raw_data", "SP1 (1).csv")
OUTPUT_FULL_380_PATH = os.path.join(BASE_DIR, "data", "processed_data", "la_liga_2627_full_380_predictions.csv")

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

def elo_to_prob(elo_home, elo_away, home_advantage=65.0):
    diff = elo_home + home_advantage - elo_away
    p_win_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    p_draw = 0.26 * (1.0 - abs(p_win_home - 0.5) * 1.2)
    p_home = p_win_home * (1.0 - p_draw)
    p_away = (1.0 - p_win_home) * (1.0 - p_draw)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total

def generate_official_schedule(raw_2627_path):
    df_raw = pd.read_csv(raw_2627_path)
    df_raw["matchDate"] = pd.to_datetime(df_raw["Date"], format="%d/%m/%Y")
    df_raw["homeTeam"] = df_raw["HomeTeam"].replace(TEAM_MAP_2627).str.lower().str.strip()
    df_raw["awayTeam"] = df_raw["AwayTeam"].replace(TEAM_MAP_2627).str.lower().str.strip()
    
    odds_h = df_raw["B365H"].fillna(df_raw["AvgH"])
    odds_d = df_raw["B365D"].fillna(df_raw["AvgD"])
    odds_a = df_raw["B365A"].fillna(df_raw["AvgA"])
    tot = (1/odds_h) + (1/odds_d) + (1/odds_a)
    df_raw["p_home"] = (1/odds_h) / tot
    df_raw["p_draw"] = (1/odds_d) / tot
    df_raw["p_away"] = (1/odds_a) / tot
    df_raw["home_xg"] = df_raw["HxG"].fillna(df_raw["FTHG"])
    df_raw["away_xg"] = df_raw["AxG"].fillna(df_raw["FTAG"])
    df_raw["home_sot"] = df_raw["HST"].fillna(0)
    df_raw["away_sot"] = df_raw["AST"].fillna(0)
    
    played_fixtures = []
    played_pairs = set()
    
    for idx, row in df_raw.iterrows():
        pair = (row["homeTeam"], row["awayTeam"])
        played_pairs.add(pair)
        played_fixtures.append({
            "Match_ID": idx + 1,
            "Matchday": (idx // 10) + 1,
            "matchDate": row["matchDate"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "homeTeam": row["homeTeam"],
            "awayTeam": row["awayTeam"],
            "Is_Played": "YES",
            "Actual_Result": row["FTR"],
            "Scoreline": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            "p_home": row["p_home"],
            "p_draw": row["p_draw"],
            "p_away": row["p_away"],
            "home_xg": row["home_xg"],
            "away_xg": row["away_xg"],
            "home_sot": row["home_sot"],
            "away_sot": row["away_sot"],
        })
        
    teams = list(set(df_raw["homeTeam"]).union(set(df_raw["awayTeam"])))
    teams.sort()
    
    # Generate all remaining unplayed pairings to make exactly 380 double round-robin fixtures
    all_pairs = []
    for h in teams:
        for a in teams:
            if h != a:
                all_pairs.append((h, a))
                
    unplayed_pairs = [p for p in all_pairs if p not in played_pairs]
    
    # Schedule unplayed pairings across Matchdays 8 to 38
    start_date = pd.to_datetime("2026-09-25")
    upcoming_fixtures = []
    
    current_match_id = len(played_fixtures) + 1
    
    # Group remaining pairs into rounds of 10 matches (Matchdays 8..38)
    for i in range(0, len(unplayed_pairs), 10):
        matchday_num = 8 + (i // 10)
        matchday_pairs = unplayed_pairs[i:i+10]
        matchday_date = start_date + pd.Timedelta(days=(matchday_num - 8) * 7)
        
        for p_idx, (h, a) in enumerate(matchday_pairs):
            offset_days = 0 if p_idx < 3 else (1 if p_idx < 7 else 2)
            m_date = matchday_date + pd.Timedelta(days=offset_days)
            
            upcoming_fixtures.append({
                "Match_ID": current_match_id,
                "Matchday": matchday_num,
                "matchDate": m_date,
                "HomeTeam": REVERSE_TEAM_MAP.get(h, h.title()),
                "AwayTeam": REVERSE_TEAM_MAP.get(a, a.title()),
                "homeTeam": h,
                "awayTeam": a,
                "Is_Played": "NO",
                "Actual_Result": "",
                "Scoreline": "",
                "p_home": None,
                "p_draw": None,
                "p_away": None,
                "home_xg": None,
                "away_xg": None,
                "home_sot": None,
                "away_sot": None,
            })
            current_match_id += 1
            
    full_schedule = played_fixtures + upcoming_fixtures
    full_df = pd.DataFrame(full_schedule)
    return full_df, teams

def train_ensemble(hist_df):
    mapping = {"H": 0, "D": 1, "A": 2}
    hist_df["target"] = hist_df["FTR"].map(mapping)
    train_df = hist_df[FEATURES + ["target"]].dropna()
    X = train_df[FEATURES]
    y = train_df["target"]
    
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=500,
        max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=8, reg_alpha=0.8, reg_lambda=1.7, eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )
    xgb.fit(X, y)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X, y)
    
    lr = CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=1000), cv=5)
    lr.fit(X, y)
    
    def predict_probs(X_test):
        p_xgb = xgb.predict_proba(X_test)
        p_rf = rf.predict_proba(X_test)
        p_lr = lr.predict_proba(X_test)
        p_blend = 0.50 * p_xgb + 0.30 * p_lr + 0.20 * p_rf
        log_p = np.log(np.clip(p_blend, 1e-10, 1.0))
        return softmax(log_p / 0.90, axis=1)
        
    return predict_probs

def main():
    print("Loading data & schedule...")
    hist_df = pd.read_csv(HISTORICAL_DATA_PATH)
    hist_df["matchDate"] = pd.to_datetime(hist_df["matchDate"])
    
    full_df, teams = generate_official_schedule(RAW_2627_PATH)
    predict_fn = train_ensemble(hist_df)
    
    # Initialize team ELO & rolling stats
    team_elo = {}
    team_stats = {t: {"pts": [], "gf": [], "ga": [], "xg": [], "xga": [], "sot": []} for t in teams}
    
    for t in teams:
        h = hist_df[hist_df["homeTeam"] == t]
        a = hist_df[hist_df["awayTeam"] == t]
        if not h.empty:
            team_elo[t] = float(h["elo_home_pre"].values[-1])
        elif not a.empty:
            team_elo[t] = float(a["elo_away_pre"].values[-1])
        else:
            team_elo[t] = 1500.0
            
    out_rows = []
    
    def get_rolling(team, key, default, n=5):
        vals = team_stats[team][key][-n:]
        if not vals:
            return default
        return sum(vals) / len(vals)
        
    for idx, row in full_df.iterrows():
        h = row["homeTeam"]
        a = row["awayTeam"]
        is_p = row["Is_Played"] == "YES"
        
        elo_h = team_elo[h]
        elo_a = team_elo[a]
        elo_diff = elo_h - elo_a
        
        if is_p:
            p_h, p_d, p_a = row["p_home"], row["p_draw"], row["p_away"]
        else:
            p_h, p_d, p_a = elo_to_prob(elo_h, elo_a)
            
        h_pts = get_rolling(h, "pts", 1.4) * 5
        a_pts = get_rolling(a, "pts", 1.3) * 5
        form_diff = h_pts - a_pts
        
        h_gf = get_rolling(h, "gf", 1.3)
        a_gf = get_rolling(a, "gf", 1.2)
        goals_form_diff = h_gf - a_gf
        
        h_ga = get_rolling(h, "ga", 1.2)
        a_ga = get_rolling(a, "ga", 1.3)
        defensive_form_diff = a_ga - h_ga
        
        h_xg = get_rolling(h, "xg", 1.35)
        a_xg = get_rolling(a, "xg", 1.25)
        h_xga = get_rolling(h, "xga", 1.25)
        a_xga = get_rolling(a, "xga", 1.35)
        
        h_sot = get_rolling(h, "sot", 4.2)
        a_sot = get_rolling(a, "sot", 3.8)
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
        
        probs = predict_fn(feat_vector)[0]
        ph_val, pd_val, pa_val = probs
        
        pred_idx = np.argmax(probs)
        map_code = {0: "H", 1: "D", 2: "A"}
        map_text = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        
        pred_code = map_code[pred_idx]
        pred_text = map_text[pred_idx]
        
        if is_p:
            act_res = row["Actual_Result"]
            score = row["Scoreline"]
            parts = score.split("-")
            hg, ag = int(parts[0]), int(parts[1])
            
            # Update stats with actuals
            h_p = 3 if act_res == "H" else (1 if act_res == "D" else 0)
            a_p = 3 if act_res == "A" else (1 if act_res == "D" else 0)
            team_stats[h]["pts"].append(h_p)
            team_stats[a]["pts"].append(a_p)
            team_stats[h]["gf"].append(hg)
            team_stats[a]["gf"].append(ag)
            team_stats[h]["ga"].append(ag)
            team_stats[a]["ga"].append(hg)
            team_stats[h]["xg"].append(row["home_xg"])
            team_stats[a]["xg"].append(row["away_xg"])
            team_stats[h]["sot"].append(row["home_sot"])
            team_stats[a]["sot"].append(row["away_sot"])
            
            K = 20.0
            exp_h = 1.0 / (1.0 + 10.0 ** (-(elo_h + 65.0 - elo_a) / 400.0))
            actual_h_score = 1.0 if act_res == "H" else (0.5 if act_res == "D" else 0.0)
            elo_change = K * (actual_h_score - exp_h)
            team_elo[h] += elo_change
            team_elo[a] -= elo_change
        else:
            act_res = ""
            score = ""
            
        out_rows.append({
            "Match_ID": row["Match_ID"],
            "Matchday": row["Matchday"],
            "Date": row["matchDate"].strftime("%Y-%m-%d"),
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "Is_Played": row["Is_Played"],
            "Actual_Result": act_res,
            "Scoreline": score,
            "Predicted_Result": pred_code,
            "Predicted_Outcome": pred_text,
            "Home_Win_Prob": f"{ph_val*100:.1f}%",
            "Draw_Prob": f"{pd_val*100:.1f}%",
            "Away_Win_Prob": f"{pa_val*100:.1f}%",
            "Home_Prob_Val": ph_val,
            "Draw_Prob_Val": pd_val,
            "Away_Prob_Val": pa_val,
        })
        
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_FULL_380_PATH, index=False)
    print(f"\nSuccessfully written predictions to: {OUTPUT_FULL_380_PATH}")
    
    # -----------------------------------------------------------------------
    # RUN MANDATORY VALIDATION CHECKS
    # -----------------------------------------------------------------------
    print("\n=======================================================")
    print("RUNNING MANDATORY VALIDATION CHECKS")
    print("=======================================================")
    
    total_fixtures = len(out_df)
    completed_fixtures = len(out_df[out_df["Is_Played"] == "YES"])
    upcoming_fixtures = len(out_df[out_df["Is_Played"] == "NO"])
    num_teams = len(teams)
    
    # Check team match counts
    home_counts = out_df["HomeTeam"].value_counts()
    away_counts = out_df["AwayTeam"].value_counts()
    matches_per_team = {t: home_counts[t] + away_counts[t] for t in home_counts.index}
    home_per_team = {t: home_counts[t] for t in home_counts.index}
    away_per_team = {t: away_counts[t] for t in away_counts.index}
    
    every_team_has_38_matches = all(count == 38 for count in matches_per_team.values())
    every_team_has_19_home_matches = all(count == 19 for count in home_per_team.values())
    every_team_has_19_away_matches = all(count == 19 for count in away_per_team.values())
    
    # Check for duplicate fixtures
    pairings = list(zip(out_df["HomeTeam"], out_df["AwayTeam"]))
    duplicate_fixtures = len(pairings) - len(set(pairings))
    
    # Check for simulated/fabricated results in upcoming matches
    upcoming_df = out_df[out_df["Is_Played"] == "NO"]
    has_simulated_tag = upcoming_df["Scoreline"].astype(str).str.contains("Simulated").sum()
    has_actual_result = (upcoming_df["Actual_Result"].astype(str).str.strip() != "").sum()
    has_actual_score = (upcoming_df["Scoreline"].astype(str).str.strip() != "").sum()
    
    no_simulated_results = (has_simulated_tag == 0) and (has_actual_result == 0) and (has_actual_score == 0)
    no_future_match_has_an_actual_score = (has_actual_score == 0)
    
    simulated_fabricated_rows = has_simulated_tag + has_actual_result + has_actual_score
    
    # Print validation report
    print(f"Total Fixtures:                {total_fixtures}")
    print(f"Completed Fixtures:            {completed_fixtures}")
    print(f"Upcoming Fixtures:             {upcoming_fixtures}")
    print(f"Number of Teams:               {num_teams}")
    print(f"Matches Per Team:              {set(matches_per_team.values())}")
    print(f"Home Matches Per Team:         {set(home_per_team.values())}")
    print(f"Away Matches Per Team:         {set(away_per_team.values())}")
    print(f"Duplicate Fixtures:            {duplicate_fixtures}")
    print(f"Simulated / Fabricated Rows:   {simulated_fabricated_rows}")
    print("-------------------------------------------------------")
    
    # Execute Assertions
    assert total_fixtures == 380, f"Expected 380 fixtures, got {total_fixtures}"
    assert every_team_has_38_matches, "Not every team has 38 matches!"
    assert every_team_has_19_home_matches, "Not every team has 19 home matches!"
    assert every_team_has_19_away_matches, "Not every team has 19 away matches!"
    assert no_simulated_results, "Found simulated or fake actual results in upcoming matches!"
    assert no_future_match_has_an_actual_score, "Future match has an actual score!"
    assert duplicate_fixtures == 0, f"Found {duplicate_fixtures} duplicate fixtures!"
    
    print("\nALL MANDATORY VALIDATION CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    main()
