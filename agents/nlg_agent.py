from typing import List

def generate_reasoning(players: List[str], predictions: List[float], news: str) -> str:
    top = sorted(zip(players, predictions), key=lambda x: x[1], reverse=True)[:3]
    picks = ", ".join([f"{p} ({pts:.2f})" for p, pts in top])
    note = f" | News: {news}" if news else ""
    return f"Top suggestions based on predicted points: {picks}{note}"