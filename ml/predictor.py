import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from ml.predictors.base import get_predictor
import ml.predictors.rf

def _get_supabase():
    load_dotenv()
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def _build_feature_df():
    sb = _get_supabase()
    players = pd.DataFrame(sb.table("players").select("*").execute().data)
    teams = pd.DataFrame(sb.table("teams").select("id,strength").execute().data)
    df = players.merge(teams, left_on="team_id", right_on="id", suffixes=("", "_team"))
    return df

def predict_points(player_names: list[str], model_name: str = "random_forest", model_path: str = "ml/models/rf_points.joblib") -> list[float]:
    df = _build_feature_df()
    PredictorCls = get_predictor(model_name)
    predictor = PredictorCls()
    predictor.load(model_path)
    preds = predictor.predict(df)
    by_name = {str(n).lower(): p for n, p in zip(df.get("web_name", []), preds)}
    return [by_name.get(p.lower(), 0.0) for p in player_names]
