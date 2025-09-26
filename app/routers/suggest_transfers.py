from fastapi import APIRouter
from app.schemas.team import TeamRequest
from ml.predictor import predict_points
from agents.news_agent import fetch_latest_news
from agents.nlg_agent import generate_reasoning

router = APIRouter(prefix="/fpl", tags=["FPL Suggestions"])

@router.post("/suggest_transfers")
def suggest_transfers(request: TeamRequest):
    predictions = predict_points(request.players)
    news = fetch_latest_news()
    reasoning = generate_reasoning(request.players, predictions, news)

    return {
        "suggestion": "Sell PlayerX → Buy PlayerY",
        "predictions": predictions,
        "news": news,
        "reasoning": reasoning
    }
