import logging

logger = logging.getLogger("response_tool")

class ResponseTool:
    """A tool to format and return responses."""
    def __init__(self, llm):
        self.llm = llm

    async def compose(self, user_message: str, results: dict) -> str:
        prompt = f"""
            User asked: "{user_message}"

            Here are the agent outputs:
            {results}

            Please combine these into a single, natural FPL advice message.
            """

        logger.info(f"Generated prompt for LLM: {prompt}")
        try:
            response = await self.llm.generate(prompt)
            logger.info(f"LLM response: {response}")
        except Exception as e:
            logger.error(f"Error occurred while generating LLM response: {e}")
            response = "I'm sorry, but I couldn't generate a response at this time."
            
        return response