from fastapi import FastAPI
from schemas.team import TeamRequest
from ml.predictor import suggest_transfer
from agents.news_agent import fetch_latest_news
from agents.nlg_agent import generate_reasoning

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FPL Agent is running!"}

@app.post("/suggest-transfer")
def transfer_endpoint(request: TeamRequest):
    predictions = suggest_transfer(request.players)
    news = fetch_latest_news()
    reasoning = generate_reasoning(request.players, predictions, news)
    return {"predictions": predictions, "reasoning": reasoning}
