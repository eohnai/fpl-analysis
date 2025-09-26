import os
from supabase import create_client, Client
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
import numpy as np
import requests

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
FPL_BASE_URL = 'https://fantasy.premierleague.com/api/'
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FETCH DATA FROM SUPABASE ---
players_data = supabase.table("players").select("*").execute().data
teams_data = supabase.table("teams").select("*").execute().data

# --- CONVERT TO DATAFRAMES ---
players = pd.DataFrame(players_data)
teams = pd.DataFrame(teams_data)

# --- MERGE TEAM FEATURES INTO PLAYER DATA ---
players = players.merge(
    teams[['id', 'strength']],
    left_on='team_id', right_on='id',
    suffixes=('', '_team')
)

# --- SELECT FEATURES AND TARGET ---
features = [
    'goals_scored', 'assists', 'minutes', 'form', 'selected_by_percent',
    'transfers_in', 'transfers_out', 'now_cost', 'element_type', 'strength'
]
target = 'total_points'

X = players[features]
y = players[target]

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAIN MODEL ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- EVALUATE MODEL ---
y_pred = model.predict(X_test)
print(f"R^2 on test set: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE on test set: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# --- FEATURE IMPORTANCE ---
importances = pd.Series(model.feature_importances_, index=features)
print("\nFeature importances:")
print(importances.sort_values(ascending=False))

# --- VALUE ANALYSIS: Identify undervalued players ---
# Predict points for all players in the dataset
players['predicted_points'] = model.predict(X)
# Calculate difference between actual and predicted points
players['value_diff'] = players['total_points'] - players['predicted_points']

# Show top 25 most undervalued players (actual > predicted)
print("\nTop 25 undervalued players (actual points > predicted):")
print(players[['web_name', 'total_points', 'predicted_points', 'value_diff']]
      .sort_values('value_diff', ascending=False)
      .head(25))

# --- 5-GW EXPECTED POINTS PROJECTION (heuristic) ---
try:
    def get_fpl(endpoint: str):
        return requests.get(FPL_BASE_URL + endpoint, timeout=30).json()

    # Load fixtures and events to determine next 5 gameweeks
    bootstrap = get_fpl("bootstrap-static")
    events = pd.DataFrame(bootstrap["events"]) if "events" in bootstrap else pd.DataFrame()
    if not events.empty and events["is_next"].any():
        next_event = int(events.loc[events["is_next"] == True, "id"].iloc[0])
    else:
        # Fallback: choose next after current, else max+1
        if not events.empty and events["is_current"].any():
            next_event = int(events.loc[events["is_current"] == True, "id"].max() + 1)
        else:
            next_event = 1

    target_events = set(range(next_event, next_event + 5))

    fixtures = get_fpl("fixtures")
    fixtures_df = pd.DataFrame(fixtures)
    # Keep only fixtures with an assigned event (GW) that falls within our window
    fixtures_df = fixtures_df[fixtures_df["event"].isin(target_events)]

    # Ensure we have team strengths
    teams_strength = teams.set_index("id")["strength"].to_dict()

    # Prepare players baseline per-game points
    # Use FPL points_per_game if available; otherwise approximate from total_points and minutes
    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    baseline_ppg = players.get("points_per_game", pd.Series(index=players.index)).apply(safe_float)
    fallback_ppg = players["total_points"] / np.maximum(players["minutes"] / 90.0, 1.0)
    players["baseline_ppg"] = baseline_ppg.fillna(fallback_ppg).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Heuristic adjustment per fixture (offensive tilt):
    # - Home advantage: 1.05 home, 0.95 away
    # - Opponent strength adjustment: scale by opponent_strength around league mean
    #   adj = 1 - k * ((opp_strength - mean_strength) / (max_strength - min_strength))
    #   where k=0.35
    if len(teams_strength) > 0 and not fixtures_df.empty:
        strengths = np.array(list(teams_strength.values()))
        mean_s = strengths.mean()
        min_s, max_s = strengths.min(), strengths.max()
        denom = max(1.0, (max_s - min_s))
        k = 0.35

        # Build per-team list of upcoming fixture adjustments
        team_to_adj = {}
        for _, fx in fixtures_df.iterrows():
            team_h = int(fx["team_h"]) if "team_h" in fx and pd.notna(fx["team_h"]) else None
            team_a = int(fx["team_a"]) if "team_a" in fx and pd.notna(fx["team_a"]) else None
            if team_h is None or team_a is None:
                continue
            opp_strength_for_home = teams_strength.get(team_a, mean_s)
            opp_strength_for_away = teams_strength.get(team_h, mean_s)

            opp_term_home = (opp_strength_for_home - mean_s) / denom
            opp_term_away = (opp_strength_for_away - mean_s) / denom

            adj_home = (1.05) * (1 - k * opp_term_home)
            adj_away = (0.95) * (1 - k * opp_term_away)

            team_to_adj.setdefault(team_h, []).append(adj_home)
            team_to_adj.setdefault(team_a, []).append(adj_away)

        # Aggregate expected points over next 5 GWs by summing baseline_ppg * adj for each upcoming fixture
        def project_player(row):
            team_id = int(row["team_id"]) if pd.notna(row["team_id"]) else None
            if team_id is None:
                return 0.0
            adjs = team_to_adj.get(team_id, [])
            if not adjs:
                return 0.0
            return float(row["baseline_ppg"]) * float(np.sum(adjs))

        players["xPts_next_5gw"] = players.apply(project_player, axis=1)
    else:
        players["xPts_next_5gw"] = 0.0

    # Map team_id to team_name for nicer output
    team_name_map = teams.set_index("id")["name"].to_dict()

    # Add expected/actual goal involvements and delta
    # Use season totals from bootstrap-static if available on players table; otherwise use columns from players df
    # players may already include expected_goal_involvements (xGI) and goals+assists
    if "expected_goal_involvements" in players.columns:
        players["season_xgi"] = pd.to_numeric(players["expected_goal_involvements"], errors="coerce").fillna(0.0)
    else:
        players["season_xgi"] = 0.0

    # Actual involvements: goals + assists (season to date)
    if {"goals_scored", "assists"}.issubset(players.columns):
        players["season_gi"] = players["goals_scored"].fillna(0) + players["assists"].fillna(0)
    else:
        players["season_gi"] = 0.0

    players["gi_delta"] = players["season_gi"] - players["season_xgi"]

    # Points: xPts vs actual so far and delta
    # xPts so far approximated as baseline_ppg * (minutes/90)
    players["xPts_so_far"] = players["baseline_ppg"] * np.maximum(players["minutes"] / 90.0, 0.0)
    players["actual_points_so_far"] = players["total_points"].fillna(0.0)
    players["points_delta_so_far"] = players["actual_points_so_far"] - players["xPts_so_far"]

    # Output top 10 per position
    position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players["position"] = players["element_type"].map(position_map)
    players["team_name"] = players["team_id"].map(team_name_map)

    print("\nTop 10 xPts over next 5 GWs by position:")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        top10 = (players[players["position"] == pos]
                 .sort_values("xPts_next_5gw", ascending=False)
                 .head(10)[["web_name", "position", "team_name", "minutes", "baseline_ppg", "xPts_next_5gw", "season_xgi", "season_gi", "gi_delta", "xPts_so_far", "actual_points_so_far", "points_delta_so_far"]])
        print(f"\n[{pos}]\n{top10.to_string(index=False)}")

except Exception as e:
    print(f"\nFailed to compute 5-GW expected points projection: {e}")

# --- NEXT STEPS ---
# - Add more features (from fixtures, etc.)
# - Try other models (XGBoost, LightGBM)
# - Use the trained model to predict for new players or upcoming GWs 