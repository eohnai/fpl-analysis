from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class TeamRequest(BaseModel):
    players: List[str]
    bank: float
    free_transfers: int
    chips: Dict[str, bool] = Field(default_factory=dict)
    captain: Optional[str] = None
    vice_captain: Optional[str] = None