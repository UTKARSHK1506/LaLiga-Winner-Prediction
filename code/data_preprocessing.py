import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data")

FIFA_FILES = [
    "fifa15.csv", "fifa16.csv", "fifa17.csv", "fifa18.csv",
    "fifa19.csv", "fifa20.csv", "fifa21.csv", "fifa22.csv",
    "fifa23.csv", "fifa24.csv", "fifa25.csv",
]

TEAM_MAP_MAIN = {
    "atletico madrid": "ath madrid",
    "atl. madrid":     "ath madrid",
    "deportivo la coruna": "la coruna",
    "dep. la coruna":  "la coruna",
    "sporting gijon":  "sp gijon",
    "gijon":           "sp gijon",
    "rayo vallecano":  "vallecano",
    "espanyol":        "espanol",
    "real sociedad":   "sociedad",
    "celta vigo":      "celta",
    "cadiz cf":        "cadiz",
    "granada cf":      "granada",
}

TEAM_MAP_FIFA = {
    "Atlético Madrid":           "ath madrid",
    "Athletic Club":             "ath bilbao",
    "FC Barcelona":              "barcelona",
    "Real Betis Balompié":       "betis",
    "Sevilla FC":                "sevilla",
    "Valencia CF":               "valencia",
    "Villarreal CF":             "villarreal",
    "Real Sociedad":             "sociedad",
    "RC Celta":                  "celta",
    "RCD Espanyol":              "espanol",
    "Getafe CF":                 "getafe",
    "Levante UD":                "levante",
    "Málaga CF":                 "malaga",
    "CA Osasuna":                "osasuna",
    "Girona FC":                 "girona",
    "Elche CF":                  "elche",
    "Deportivo Alavés":          "alaves",
    "UD Las Palmas":             "las palmas",
    "CD Leganés":                "leganes",
    "SD Eibar":                  "eibar",
    "Real Valladolid CF":        "valladolid",
    "UD Almería":                "almeria",
    "RCD Mallorca":              "mallorca",
    "Real Sporting de Gijón":    "sp gijon",
    "RC Deportivo de La Coruña": "la coruna",
    "Cádiz CF":                  "cadiz",
    "Granada CF":                "granada",
    "SD Huesca":                 "huesca",
    "Rayo Vallecano":            "vallecano",
    "Real Madrid":               "real madrid",
    "Córdoba CF":                None,
}

KEEP_COLS = [
    "matchDate", "Season", "homeTeam", "awayTeam",
    "FTHG", "FTAG", "FTR",
    "1xbh", "1xbd", "1xba",
    "HBPFT", "ABPFT",
    "HTSFT", "ATSFT", "HSONFT", "ASONFT", "HSOFFFT", "ASOFFFT",
    "home_xg_name", "away_xg_name",
    "home_xg", "away_xg",
    "home_ppda", "away_ppda",
    "home_deep", "away_deep",
    "home_xpts", "away_xpts",
]


def load_main(path):
    df = pd.read_csv(path)
    df = df[KEEP_COLS].copy()
    df["homeTeam"] = df["homeTeam"].str.lower().str.strip().replace(TEAM_MAP_MAIN)
    df["awayTeam"] = df["awayTeam"].str.lower().str.strip().replace(TEAM_MAP_MAIN)
    return df


def load_elo(path):
    elo = pd.read_csv(path, low_memory=False)
    elo = elo[elo["Division"] == "SP1"].copy()
    elo = elo.rename(columns={
        "HomeTeam": "homeTeam", "AwayTeam": "awayTeam",
        "HomeElo": "elo_home_pre", "AwayElo": "elo_away_pre",
    })
    elo["homeTeam"] = elo["homeTeam"].str.lower().str.strip()
    elo["awayTeam"] = elo["awayTeam"].str.lower().str.strip()
    elo["season"] = pd.to_datetime(elo["MatchDate"]).apply(
        lambda x: f"{x.year-1}/{x.year}" if x.month < 8 else f"{x.year}/{x.year+1}"
    )
    return elo[["season", "homeTeam", "awayTeam", "elo_home_pre", "elo_away_pre"]]


def load_fifa(data_dir, filenames):
    dfs = []
    for fname in filenames:
        d = pd.read_csv(os.path.join(data_dir, fname))
        d["source_file"] = fname
        dfs.append(d)
    fifa = pd.concat(dfs, ignore_index=True)
    fifa.columns = fifa.columns.str.strip().str.replace("\t", "", regex=False)
    fifa = fifa[["Name", "Overall", "Attack", "Midfield", "Defence", "source_file"]].copy()
    fifa.columns = ["team", "overall", "attack", "midfield", "defence", "source_file"]
    fifa["Season"] = fifa["source_file"].apply(
        lambda f: f"{2000 + int(f[4:6]) - 1}/{2000 + int(f[4:6])}"
    )
    fifa["team"] = (
        fifa["team"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("La Liga", "", regex=False)
        .str.strip()
    )
    for col in ["overall", "attack", "midfield", "defence"]:
        fifa[col] = pd.to_numeric(fifa[col], errors="coerce")
    fifa["team"] = fifa["team"].replace(TEAM_MAP_FIFA)
    fifa = fifa.dropna(subset=["team", "overall"]).drop(columns=["source_file"])
    return fifa


def merge_elo(df, elo):
    return df.merge(
        elo,
        left_on=["Season", "homeTeam", "awayTeam"],
        right_on=["season", "homeTeam", "awayTeam"],
        how="left",
    ).drop(columns=["season"])


def merge_fifa(df, fifa):
    df = df.merge(
        fifa.rename(columns={
            "team": "homeTeam", "overall": "overall_home",
            "attack": "attack_home", "midfield": "midfield_home", "defence": "defence_home",
        }),
        on=["homeTeam", "Season"], how="left",
    )
    df = df.merge(
        fifa.rename(columns={
            "team": "awayTeam", "overall": "overall_away",
            "attack": "attack_away", "midfield": "midfield_away", "defence": "defence_away",
        }),
        on=["awayTeam", "Season"], how="left",
    )
    return df


def compute_rolling_xg(df, window=5):
    df["matchDate"] = pd.to_datetime(df["matchDate"])

    home = df[["matchDate", "homeTeam", "awayTeam", "FTHG", "FTAG",
               "home_xg", "away_xg", "HTSFT", "HSONFT"]].copy()
    home = home.rename(columns={
        "matchDate": "date", "homeTeam": "team", "awayTeam": "opponent",
        "FTHG": "goals_for", "FTAG": "goals_against",
        "home_xg": "xg_for", "away_xg": "xg_against",
        "HTSFT": "shots", "HSONFT": "sot",
    })
    home["is_home"] = 1

    away = df[["matchDate", "awayTeam", "homeTeam", "FTAG", "FTHG",
               "away_xg", "home_xg", "ATSFT", "ASONFT"]].copy()
    away = away.rename(columns={
        "matchDate": "date", "awayTeam": "team", "homeTeam": "opponent",
        "FTAG": "goals_for", "FTHG": "goals_against",
        "away_xg": "xg_for", "home_xg": "xg_against",
        "ATSFT": "shots", "ASONFT": "sot",
    })
    away["is_home"] = 0

    long = pd.concat([home, away]).sort_values(["team", "date"])

    for raw, avg in [
        ("xg_for",     "xg_for_avg"),
        ("xg_against", "xg_against_avg"),
        ("goals_for",  "goals_for_avg"),
        ("sot",        "sot_avg"),
    ]:
        long[avg] = long.groupby("team")[raw].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

    home_f = long[long["is_home"] == 1][["date", "team", "xg_for_avg", "xg_against_avg", "goals_for_avg", "sot_avg"]]
    away_f = long[long["is_home"] == 0][["date", "team", "xg_for_avg", "xg_against_avg", "goals_for_avg", "sot_avg"]]

    df = df.merge(
        home_f.rename(columns={
            "date": "matchDate", "team": "homeTeam",
            "xg_for_avg": "home_xg_avg", "xg_against_avg": "home_xga_avg",
            "goals_for_avg": "home_goals_avg", "sot_avg": "home_sot_avg",
        }),
        on=["matchDate", "homeTeam"], how="left",
    )
    df = df.merge(
        away_f.rename(columns={
            "date": "matchDate", "team": "awayTeam",
            "xg_for_avg": "away_xg_avg", "xg_against_avg": "away_xga_avg",
            "goals_for_avg": "away_goals_avg", "sot_avg": "away_sot_avg",
        }),
        on=["matchDate", "awayTeam"], how="left",
    )
    df = df.drop(columns=[
        "home_xg", "away_xg",
        "HTSFT", "ATSFT", "HSONFT", "ASONFT", "HSOFFFT", "ASOFFFT",
    ], errors="ignore")
    df = df.dropna(subset=["home_xg_avg", "away_xg_avg"])
    return df


def compute_rolling_form(df, window=5):
    # Exactly as in EDA_point_2 cells 5-9
    def get_points(row):
        if row["FTR"] == "H": return 3, 0
        elif row["FTR"] == "A": return 0, 3
        else: return 1, 1

    df[["home_points", "away_points"]] = df.apply(
        lambda row: pd.Series(get_points(row)), axis=1
    )

    home_df = df[["matchDate", "homeTeam", "home_points", "FTHG", "FTAG"]].rename(columns={
        "homeTeam": "team", "home_points": "points",
        "FTHG": "goals_for", "FTAG": "goals_against",
    })
    away_df = df[["matchDate", "awayTeam", "away_points", "FTAG", "FTHG"]].rename(columns={
        "awayTeam": "team", "away_points": "points",
        "FTAG": "goals_for", "FTHG": "goals_against",
    })
    team_df = pd.concat([home_df, away_df])
    team_df = team_df.sort_values(["team", "matchDate"])

    team_df["points_last5"] = (
        team_df.groupby("team")["points"]
        .transform(lambda x: x.shift(1).rolling(window).sum())
    )
    team_df["goals_for_last5"] = (
        team_df.groupby("team")["goals_for"]
        .transform(lambda x: x.shift(1).rolling(window).sum())
    )
    team_df["goals_against_last5"] = (
        team_df.groupby("team")["goals_against"]
        .transform(lambda x: x.shift(1).rolling(window).sum())
    )

    home_form = team_df.rename(columns={
        "team": "homeTeam",
        "points_last5": "home_points_last5",
        "goals_for_last5": "home_goals_last5",
        "goals_against_last5": "home_conceded_last5",
    })
    away_form = team_df.rename(columns={
        "team": "awayTeam",
        "points_last5": "away_points_last5",
        "goals_for_last5": "away_goals_last5",
        "goals_against_last5": "away_conceded_last5",
    })

    df = df.merge(
        home_form[["matchDate", "homeTeam", "home_points_last5", "home_goals_last5", "home_conceded_last5"]],
        on=["matchDate", "homeTeam"], how="left",
    )
    df = df.merge(
        away_form[["matchDate", "awayTeam", "away_points_last5", "away_goals_last5", "away_conceded_last5"]],
        on=["matchDate", "awayTeam"], how="left",
    )

    df.fillna(0, inplace=True)
    df["form_diff"]           = df["home_points_last5"] - df["away_points_last5"]
    df["goals_form_diff"]     = df["home_goals_last5"]  - df["away_goals_last5"]
    df["defensive_form_diff"] = df["away_conceded_last5"] - df["home_conceded_last5"]

    return df


def compute_rest_days(df):
    # Exactly as in EDA.ipynb cells 60-65
    df_long = pd.concat([
        df[["Season", "matchDate", "homeTeam"]].rename(columns={"homeTeam": "team"}),
        df[["Season", "matchDate", "awayTeam"]].rename(columns={"awayTeam": "team"}),
    ])
    df_long = df_long.sort_values(["Season", "team", "matchDate"])
    df_long["last_match"] = df_long.groupby(["Season", "team"])["matchDate"].shift(1)

    last_match_dict = {
        (row["Season"], row["team"], row["matchDate"]): row["last_match"]
        for _, row in df_long.iterrows()
    }

    df["home_last_match"] = df.apply(
        lambda row: last_match_dict.get((row["Season"], row["homeTeam"], row["matchDate"])), axis=1
    )
    df["away_last_match"] = df.apply(
        lambda row: last_match_dict.get((row["Season"], row["awayTeam"], row["matchDate"])), axis=1
    )

    df["home_rest_days"] = (df["matchDate"] - df["home_last_match"]).dt.days
    df["away_rest_days"] = (df["matchDate"] - df["away_last_match"]).dt.days
    df["home_rest_days"] = df["home_rest_days"].fillna(14)
    df["away_rest_days"] = df["away_rest_days"].fillna(14)
    df["home_rest_days"] = df["home_rest_days"].clip(1, 30)
    df["away_rest_days"] = df["away_rest_days"].clip(1, 30)
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    return df


def compute_venue_strength(df):
    home_perf = df.groupby("homeTeam")["FTR"].apply(lambda x: (x == "H").mean())
    away_perf = df.groupby("awayTeam")["FTR"].apply(lambda x: (x == "A").mean())
    df["home_team_home_strength"] = df["homeTeam"].map(home_perf)
    df["away_team_away_strength"] = df["awayTeam"].map(away_perf)
    df["venue_bias_diff"] = df["home_team_home_strength"] - df["away_team_away_strength"]
    return df


def engineer_features(df):
    df["xga_diff"]      = df["home_xga_avg"]  - df["away_xga_avg"]
    df["elo_diff"]      = df["elo_home_pre"]   - df["elo_away_pre"]
    df["overall_diff"]  = df["overall_home"]   - df["overall_away"]
    df["attack_diff"]   = df["attack_home"]    - df["attack_away"]
    df["midfield_diff"] = df["midfield_home"]  - df["midfield_away"]
    df["defence_diff"]  = df["defence_home"]   - df["defence_away"]
    df["ppda_diff"]     = df["home_ppda"]      - df["away_ppda"]
    df["deep_diff"]     = df["home_deep"]      - df["away_deep"]
    df["xpts_diff"]     = df["home_xpts"]      - df["away_xpts"]

    df["p_home_raw"] = 1 / df["1xbh"]
    df["p_draw_raw"] = 1 / df["1xbd"]
    df["p_away_raw"] = 1 / df["1xba"]
    total = df["p_home_raw"] + df["p_draw_raw"] + df["p_away_raw"]
    df["p_home"]     = df["p_home_raw"] / total
    df["p_draw"]     = df["p_draw_raw"] / total
    df["p_away"]     = df["p_away_raw"] / total
    df["overround"]  = total
    df["odds_home_away_diff"] = df["1xba"] - df["1xbh"]

    df["predicted"]  = df[["p_home", "p_draw", "p_away"]].idxmax(axis=1).map(
        {"p_home": "H", "p_draw": "D", "p_away": "A"}
    )
    df["confidence"] = df[["p_home", "p_draw", "p_away"]].max(axis=1)

    df["xg_vs_def_diff"] = df["home_xg_avg"] - df["away_xga_avg"]
    df["finishing_efficiency_diff"] = (
        (df["home_goals_avg"] - df["home_xg_avg"]) -
        (df["away_goals_avg"] - df["away_xg_avg"])
    )
    df["total_xg"]       = df["home_xg_avg"] + df["away_xg_avg"]
    df["xg_closeness"]   = (df["home_xg_avg"] - df["away_xg_avg"]).abs()
    df["draw_xg_signal"] = -df["xg_closeness"]

    df["sot_diff"]             = df["home_sot_avg"] - df["away_sot_avg"]
    df["elo_form_interaction"]  = df["elo_diff"] * df["form_diff"]

    df["year"]  = df["matchDate"].dt.year
    df["month"] = df["matchDate"].dt.month
    return df


def run(main_path, elo_path, fifa_dir, output_path):
    df   = load_main(main_path)
    elo  = load_elo(elo_path)
    fifa = load_fifa(fifa_dir, FIFA_FILES)

    df = merge_elo(df, elo)
    df = merge_fifa(df, fifa)
    df = compute_rolling_xg(df)
    df = df.sort_values("matchDate").reset_index(drop=True)
    df = compute_rolling_form(df)
    df = compute_rest_days(df)
    df = compute_venue_strength(df)
    df = engineer_features(df)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows, {len(df.columns)} columns → {output_path}")
    return df


if __name__ == "__main__":
    run(
        main_path=os.path.join(DATA_DIR, "la_liga_dataset.csv"),
        elo_path=os.path.join(DATA_DIR, "Matches.csv"),
        fifa_dir=DATA_DIR,
        output_path=os.path.join(DATA_DIR, "..", "processed_data", "la_liga_data.csv"),
    )