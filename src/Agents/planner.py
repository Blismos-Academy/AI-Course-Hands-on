from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


Route = Literal["memory", "rag", "tools", "both"]


class Plan(BaseModel):
    route: Route = Field(
        description=(
            "Execution route required to answer the user's request."
        )
    )


class Planner:
    """Determines which part of the system is required."""

    SYSTEM_PROMPT = """
You are the planning component of a trip-planning assistant.

Analyze the user's request and conversation history.

Choose exactly one route:

memory:
Use this when the request can be answered from conversation history alone.

rag:
Use this when knowledge-base information is required but live information
is not required.

tools:
Use this when live external information is required but knowledge-base
information is not required.

both:
Use this when both live external information and knowledge-base information
are required.

Do not answer the user.
Do not execute tools.
Do not use keyword matching.
Make the decision from the semantic meaning of the request.
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