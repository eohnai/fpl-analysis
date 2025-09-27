"""Base agent class for all agents."""

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    async def run(self, message: str) -> str:
        pass
