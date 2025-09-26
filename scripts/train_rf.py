import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import requests
 

FPL_BASE_URL = 'https://fantasy.premierleague.com/api/'
logger = logging.getLogger("train_rf")

def get_supabase() -> Client:
    """Get the supabase client"""
    logger.info("Loading environment variables for Supabase...")
    try:
        load_dotenv()
    except Exception as e:
        logger.error("Failed to load .env: %s", e)

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

    return create_client(SUPABASE_URL, SUPABASE_KEY)

def load_data(supabase: Client) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the data from the supabase client"""
    logger.info("Loading data from supabase...")
    try:
        players_data = supabase.table("players").select("*").execute().data
        teams_data = supabase.table("teams").select("*").execute().data

        players = pd.DataFrame(players_data)
        teams = pd.DataFrame(teams_data)
        logger.info("Loaded tables: players=%d rows, teams=%d rows", len(players), len(teams))
    except Exception as e:
        logger.error("Error loading data from supabase: %s", e)
        raise

    return players, teams

def build_dataset(players: pd.DataFrame, teams: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Build the dataset for training"""
    logger.info("Building dataset...")
    df = players.merge(teams[['id', 'strength']], left_on='team_id', right_on='id', suffixes=('', '_team'))
    features = ['goals_scored', 'assists', 'minutes', 'form', 'selected_by_percent', 
                'transfers_in', 'transfers_out', 'now_cost', 'element_type', 'strength']
    target = 'total_points'

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(df[target], errors='coerce').fillna(0)
    logger.debug("Feature matrix shape=%s, target shape=%s", X.shape, y.shape)

    return X, y, features

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int, random_state: int) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train the Random Forest model"""
    logger.info("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    logger.info("Split data: train=%d, test=%d", len(X_train), len(X_test))
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    t0 = time.time()
    model.fit(X_train, y_train)
    
    logger.info("Training completed in %.2f s", time.time() - t0)
    y_pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
    }
    
    logger.info("Evaluation metrics: R2=%.3f RMSE=%.2f", metrics["r2"], metrics["rmse"])
    return model, metrics

def save_artifacts(model: RandomForestRegressor, features: List[str], path: str) -> None:
    """Save the model and features to a joblib file"""
    logger.info("Saving model artifacts to %s...", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump({"model": model, "features": features}, path)
    
    logger.info("Saved model artifact: %s", path)

def compute_projection(players: pd.DataFrame, teams: pd.DataFrame, weeks: int) -> pd.Series:
    """Compute projection based on upcoming fixtures and team strengths"""
    def get_fpl(endpoint: str):
        return requests.get(FPL_BASE_URL + endpoint, timeout=30).json()
    try:
        logger.info("Computing %d-GW projection via FPL API", weeks)
        bootstrap = get_fpl("bootstrap-static")
        events = pd.DataFrame(bootstrap.get("events", []))
        if not events.empty and events["is_next"].any():
            next_event = int(events.loc[events["is_next"] == True, "id"].iloc[0])
        elif not events.empty and events["is_current"].any():
            next_event = int(events.loc[events["is_current"] == True, "id"].max() + 1)
        else:
            next_event = 1
        target_events = set(range(next_event, next_event + max(int(weeks), 0)))
        fixtures_df = pd.DataFrame(get_fpl("fixtures"))
        fixtures_df = fixtures_df[fixtures_df["event"].isin(target_events)]
        teams_strength = teams.set_index("id")["strength"].to_dict()

        def safe_float(x):
            try: return float(x)
            except Exception: return np.nan
        baseline_ppg = players.get("points_per_game", pd.Series(index=players.index)).apply(safe_float)
        fallback_ppg = players["total_points"] / np.maximum(players["minutes"] / 90.0, 1.0)
        players["baseline_ppg"] = baseline_ppg.fillna(fallback_ppg).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if len(teams_strength) == 0 or fixtures_df.empty:
            logger.warning("Projection skipped: missing teams_strength or fixtures for target events")
            return pd.Series(0.0, index=players.index)

        strengths = np.array(list(teams_strength.values()))
        mean_s = strengths.mean()
        min_s, max_s = strengths.min(), strengths.max()
        denom = max(1.0, (max_s - min_s)); k = 0.35

        team_to_adj: Dict[int, list] = {}
        for _, fx in fixtures_df.iterrows():
            team_h = int(fx["team_h"]) if pd.notna(fx.get("team_h")) else None
            team_a = int(fx["team_a"]) if pd.notna(fx.get("team_a")) else None
            if team_h is None or team_a is None: continue
            opp_home = teams_strength.get(team_a, mean_s)
            opp_away = teams_strength.get(team_h, mean_s)
            adj_home = (1.05) * (1 - k * ((opp_home - mean_s)/denom))
            adj_away = (0.95) * (1 - k * ((opp_away - mean_s)/denom))
            team_to_adj.setdefault(team_h, []).append(adj_home)
            team_to_adj.setdefault(team_a, []).append(adj_away)

        def proj(row):
            team_id = int(row["team_id"]) if pd.notna(row.get("team_id")) else None
            adjs = team_to_adj.get(team_id, [])
            return float(row["baseline_ppg"]) * float(np.sum(adjs)) if adjs else 0.0

        proj_series = players.apply(proj, axis=1)
        logger.info("Projection computed for %d players", len(proj_series))
        return proj_series
    except Exception:
        logger.warning("Projection failed; returning zeros", exc_info=True)
        return pd.Series(0.0, index=players.index)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger.info("Starting Random Forest training job")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--artifact_path", type=str, default="ml/models/rf_points.joblib")
    parser.add_argument("--projection_weeks", type=int, default=5)
    args = parser.parse_args()

    try:
        supabase = get_supabase()
    except Exception:
        logger.error("Supabase initialization failed", exc_info=True)
        raise SystemExit(1)

    try:
        t_load = time.time()
        players, teams = load_data(supabase)
        logger.info("Data loaded in %.2f s", time.time() - t_load)
    except Exception:
        logger.error("Data load failed", exc_info=True)
        raise SystemExit(1)

    X, y, features = build_dataset(players, teams)
    model, metrics = train_model(X, y, args.n_estimators, args.random_state)
    logger.info("Metrics summary: R2=%.3f RMSE=%.2f", metrics['r2'], metrics['rmse'])

    players = players.copy()
    players['predicted_points'] = model.predict(X)
    players['value_diff'] = players['total_points'] - players['predicted_points']

    players['xPts_next_5gw'] = compute_projection(players, teams, weeks=args.projection_weeks)

    save_artifacts(model, features, args.artifact_path)
    logger.info("Training job completed successfully")

if __name__ == "__main__":
    main()