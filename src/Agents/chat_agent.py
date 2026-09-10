import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage
)

from agents.tool_agent import weather_tools
from agents.memory_agent import MemoryAgent


load_dotenv()


class LLMCore:

    def __init__(self):

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.llm_with_tools = self.llm.bind_tools(weather_tools)

        self.tools = {
            tool.name: tool
            for tool in weather_tools
        }

        self.memory = MemoryAgent(
            session_id="default",
            max_messages=10
        )

    def ask(self, question: str) -> str:

        messages = [
            SystemMessage(
                content=(
                    "Use the date and time tool for current date or time questions. "
                    "Use weather tools for current weather and forecast questions. "
                    "Never guess current date, time, or weather."
                )
            )
        ]

        for message in self.memory.get():
            if message["role"] == "user":
                messages.append(
                    HumanMessage(content=message["content"])
                )
            else:
                messages.append(
                    SystemMessage(
                        content=f"Previous assistant response: {message['content']}"
                    )
                )

        messages.append(
            HumanMessage(content=question)
        )

        response = self.llm_with_tools.invoke(messages)

        while response.tool_calls:

            messages.append(response)

            for tool_call in response.tool_calls:

                tool = self.tools.get(tool_call["name"])

                if tool is None:
                    raise ValueError(
                        f"Tool '{tool_call['name']}' not found."
                    )

                result = tool.invoke(tool_call["args"])

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )

            response = self.llm_with_tools.invoke(messages)

        self.memory.add("user", question)
        self.memory.add("assistant", response.content)

        return response.content