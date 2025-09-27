"""This is the orchestrator agent that will call other agents as needed based on the user's Telegram message requests."""

from tools.response_tool import ResponseTool
import logging
logger = logging.getLogger("orchestrator_agent")

class OrchestratorAgent:
    def __init__(self, llm, agents: dict):
        self.llm = llm
        self.agents = agents
        self.response_tool = ResponseTool(llm)

    async def handle_request(self, message: str) -> str:
        logger.info(f"Orchestrator received message: {message}")

        # use LLM to determine which agents to invoke
        agent_selection = await self.llm.select_agent(message)
        logger.info(f"Selected agents: {agent_selection}")
        
        results = {}
    
        if "transfer" in agent_selection:
            results["transfer"] = await self.agents["transfer_agent"].run(message)
        if "captain" in agent_selection:
            results["captain"] = await self.agents["captaincy_agent"].run(message)
        if "chip" in agent_selection:
            results["chip"] = await self.agents["chip_agent"].run(message)

        logger.info(f"Orchestrator response: {results}")
        return self.response_tool.compose(message, results)