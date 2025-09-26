from fastapi import FastAPI
from app.routers import suggest_transfers

app = FastAPI()
app.include_router(suggest_transfers.router)

@app.get("/")
def root():
    return {"message": "FPL Agent is running"}
