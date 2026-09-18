from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

Route = Literal["memory", "rag", "tools", "mcp", "both"]

class Plan(BaseModel):
    route: Route = Field(
        description="Execution route required to answer the user's request."
    )

class Planner:
    """Determines which part of the system is required."""

    SYSTEM_PROMPT = """
You are the planner for a weather, clothing, and food recommendation system.

Analyze the request and history to choose ONE route:

memory: Request answered from history alone.
rag: Weather advice, food guidelines, or clothing safety rules from knowledge base.
tools: Weather queries or live operations.
mcp: Setting calendar events, reminders, or schedule alerts via MCP.
both: Requires live external weather/tools AND knowledge-base/MCP scheduling.

Do not answer the user directly.
""".strip()

    def __init__(self, llm: ChatGroq) -> None:
        self.llm = llm.with_structured_output(Plan)

    def invoke(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
    ) -> Route:
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            *(history or []),
            HumanMessage(content=question),
        ]
        result: Plan = self.llm.invoke(messages)
        return result.route