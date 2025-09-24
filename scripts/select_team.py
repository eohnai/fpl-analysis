import os
from supabase import create_client, Client
import pandas as pd
from dotenv import load_dotenv
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpStatus

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FETCH DATA FROM SUPABASE ---
players_data = supabase.table("players").select("*").execute().data
players = pd.DataFrame(players_data)

# --- SCORING COLUMN FOR SELECTION ---
# Prefer xPts over next 5 GWs, then predicted_points, then total_points
if 'xPts_next_5gw' in players.columns:
    players['selection_score'] = players['xPts_next_5gw']
elif 'predicted_points' in players.columns:
    players['selection_score'] = players['predicted_points']
else:
    players['selection_score'] = players['total_points']

# --- POSITION MAPPING ---
position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
players['position'] = players['element_type'].map(position_map)
team_map = {t['id']: t['name'] for t in supabase.table("teams").select("id,name").execute().data}
players['team_name'] = players['team_id'].map(team_map)

# --- OPTIMAL TEAM SELECTOR (FULL SQUAD, 15 PLAYERS) ---
try:
    print("\n--- Optimal 15-player Squad (using pulp) ---")
    # FPL squad constraints
    squad_size = 15
    budget = 1000
    position_limits = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    team_limit = 3
    # Create binary variables for each player
    player_vars = {i: LpVariable(f"player_{i}", cat=LpBinary) for i in players.index}
    # Problem
    prob = LpProblem("FPL_Squad_Selection", LpMaximize)
    # Objective: maximize total selection score
    prob += lpSum([player_vars[i] * players.loc[i, 'selection_score'] for i in players.index])
    # Constraint: total cost
    prob += lpSum([player_vars[i] * players.loc[i, 'now_cost'] for i in players.index]) <= budget
    # Constraint: squad size
    prob += lpSum([player_vars[i] for i in players.index]) == squad_size
    # Constraint: position limits
    for pos, limit in position_limits.items():
        prob += lpSum([player_vars[i] for i in players.index if players.loc[i, 'position'] == pos]) == limit
    # Constraint: max 3 per team
    for team_id in players['team_id'].unique():
        prob += lpSum([player_vars[i] for i in players.index if players.loc[i, 'team_id'] == team_id]) <= team_limit
    # Solve
    prob.solve()
    # Extract selected players
    selected = [i for i in players.index if player_vars[i].varValue == 1]
    squad_df = players.loc[selected]
    # Order by position: GKP, DEF, MID, FWD, and within each by selection score desc
    pos_order_map = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad_df["_pos_order"] = squad_df["position"].map(pos_order_map)
    squad_df = squad_df.sort_values(["_pos_order", "selection_score"], ascending=[True, False]).drop(columns=["_pos_order"])
    opt_cols = ['web_name', 'position', 'team_name', 'now_cost', 'minutes']
    if 'xPts_next_5gw' in squad_df.columns:
        opt_cols += ['xPts_next_5gw']
        total_opt_score = squad_df['xPts_next_5gw'].sum()
    elif 'predicted_points' in squad_df.columns:
        opt_cols += ['predicted_points']
        total_opt_score = squad_df['predicted_points'].sum()
    else:
        opt_cols += ['total_points']
        total_opt_score = squad_df['total_points'].sum()
    print(squad_df[opt_cols])
    print(f"Total cost: {squad_df['now_cost'].sum()/10:.1f}m, Total score: {total_opt_score:.1f}")
    print(f"Solver status: {LpStatus[prob.status]}")
except ImportError:
    print("\n'pulp' is not installed. Skipping optimal squad selection. To enable, run: pip install pulp")

# --- NOTE ---
# This is a template. You can further customize constraints, add bench order, captain selection, etc. 