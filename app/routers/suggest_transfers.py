from fastapi import APIRouter
from app.schemas.team import TeamRequest
from ml.predictor import predict_points
from tools.news_tool import fetch_latest_news

router = APIRouter(prefix="/fpl", tags=["FPL Suggestions"])

@router.post("/suggest_transfers")
def suggest_transfers(request: TeamRequest):
    predictions = predict_points(request.players)
    news = fetch_latest_news()

    return {
        "suggestion": "Sell PlayerX → Buy PlayerY",
        "predictions": predictions,
        "news": news
    }
