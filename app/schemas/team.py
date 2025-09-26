from pydantic import BaseModel, Field
from typing import List, Dict

class TeamRequest(BaseModel):
    players: List[str]
    captain: str
    vice_captain: str
    bank: float
    free_transfers: int
    chips: Dict[str, bool] = Field(default_factory=dict)