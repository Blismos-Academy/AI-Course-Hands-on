import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from agents.memory_agent import MemoryAgent
from agents.planner import Planner
from agents.tool_agent import ToolAgent, weather_tools
from mcp_agent.mcp_client import MCPClient
from rag.retrival import HybridRetriever
from agent_safety.safety import AgentSafety

# Suppress standard http noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "weather-ai-agent")
SYSTEM_PROMPT = (
    "You are an assistant for weather, clothing advice, food suggestions, and calendar scheduling. "
    "Use provided live tools/MCP context to answer. Keep responses concise, clear, and direct."
)


class AgentState(TypedDict, total=False):
    question: str
    history: List[BaseMessage]
    route: str
    tool_results: List[str]
    rag_results: List[str]
    answer: str

    # Safety metadata
    safety_reason: str
    pii_detected: bool
    injection_detected: bool


class Orchestrator:
    def __init__(self) -> None:
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY"),
        )

        # Safety is deliberately initialized before the graph.
        self.safety = AgentSafety()

        self.memory = MemoryAgent(max_messages=10)
        self.planner = Planner(self.llm)
        self.tools = ToolAgent(self.llm, weather_tools)
        self.mcp_client = MCPClient()
        self.retriever = HybridRetriever(threshold=0.5)

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        # FIRST NODE: nothing reaches memory, planner, tools or RAG
        # until the safety node approves/sanitizes the request.
        workflow.add_node("safety", self._safety)
        workflow.add_node("memory", self._memory)
        workflow.add_node("planner", self._planner)
        workflow.add_node("tools", self._tools)
        workflow.add_node("rag", self._rag)
        workflow.add_node("response", self._response)

        workflow.add_edge(START, "safety")

        workflow.add_conditional_edges(
            "safety",
            lambda state: "memory" if state.get("safety_reason") == "SAFE" else "response",
            {
                "memory": "memory",
                "response": "response",
            },
        )

        workflow.add_edge("memory", "planner")
        workflow.add_edge("rag", "response")
        workflow.add_edge("response", END)

        workflow.add_conditional_edges(
            "planner",
            lambda state: state["route"],
            {
                "memory": "response",
                "rag": "rag",
                "tools": "tools",
                "mcp": "tools",
                "both": "tools",
            },
        )

        workflow.add_conditional_edges(
            "tools",
            lambda state: "rag" if state.get("route") == "both" else "response",
            {"rag": "rag", "response": "response"},
        )

        return workflow.compile()

    @traceable(name="Agent Safety Node", run_type="chain")
    def _safety(self, state: AgentState) -> dict:
        """
        Security boundary.

        Important:
        - Original unsafe input is never passed to memory/planner/LLM.
        - Safe/redacted input replaces state["question"].
        - Blocked requests terminate through the response node.
        """
        result = self.safety.check(state.get("question", ""))

        if not result.allowed:
            logger.warning("Agent Safety blocked request: %s", result.reason)

            return {
                "question": "",
                "safety_reason": result.reason,
                "pii_detected": result.pii_detected,
                "injection_detected": result.injection_detected,
                "answer": self.safety.safe_message(result),
            }

        return {
            "question": result.question,
            "safety_reason": "SAFE",
            "pii_detected": result.pii_detected,
            "injection_detected": result.injection_detected,
        }

    @traceable(name="Memory Node", run_type="chain")
    def _memory(self, state: AgentState) -> dict:
        return {"history": self.memory.get_messages()}

    @traceable(name="Planner Node", run_type="chain")
    def _planner(self, state: AgentState) -> dict:
        route = self.planner.invoke(
            state["question"],
            state.get("history", []),
        )

        # Route allowlist: the planner can only select known internal routes.
        allowed_routes = {"memory", "rag", "tools", "mcp", "both"}
        if route not in allowed_routes:
            logger.warning("Planner returned invalid route: %s", route)
            route = "memory"

        return {"route": route}

    @traceable(name="Tools Node", run_type="chain")
    def _tools(self, state: AgentState) -> dict:
        results = []
        route = state.get("route")

        if route in {"tools", "both"}:
            try:
                tool_res = self.tools.invoke(
                    state["question"],
                    state.get("history", []),
                )
                results.extend(tool_res)
            except Exception as e:
                logger.error(f"Weather Tool Error: {e}")
                results.append(f"Weather Tool Error: {e}")

        if route in {"mcp", "both"}:
            try:
                event = self._extract_event_details(state["question"])

                # Only the approved internal MCP tool is callable here.
                mcp_res = asyncio.run(
                    self.mcp_client.call_tool(
                        "create_calendar_event",
                        event,
                    )
                )
                results.append(f"MCP Calendar Response: {mcp_res}")

            except Exception as e:
                logger.error(f"MCP Tool Error: {e}")
                results.append(f"MCP Error: {e}")

        return {"tool_results": results}

    def _extract_event_details(self, question: str) -> dict:
        """Turn a safe, already-filtered request into calendar arguments."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        prompt = (
            f"Current datetime: {now} (Asia/Kolkata, UTC+05:30).\n"
            "Extract a calendar event from the request as STRICT JSON only, no prose, "
            "no markdown fences, with exactly these keys:\n"
            '"title" (short string), "start_time" (ISO 8601 with +05:30 offset), '
            '"duration_minutes" (integer, default 30 if not stated).\n\n'
            f"Request: {question}"
        )

        raw = str(
            self.llm.invoke([HumanMessage(content=prompt)]).content
        ).strip()

        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

        return json.loads(raw)

    @traceable(name="RAG Node", run_type="retriever")
    def _rag(self, state: AgentState) -> dict:
        query = state["question"]

        if state.get("tool_results"):
            query += "\n\nLive context:\n" + "\n".join(
                state["tool_results"]
            )

        rag_results = self.retriever.search(query, top_k=3) or []
        return {"rag_results": rag_results}

    @traceable(name="Response Node", run_type="chain")
    def _response(self, state: AgentState) -> dict:
        # If safety blocked the request, do NOT call the LLM.
        if state.get("safety_reason") != "SAFE":
            return {"answer": state.get("answer", "Request blocked by safety policy.")}

        context_parts = []

        if state.get("tool_results"):
            context_parts.append(
                "Live context:\n" + "\n".join(state["tool_results"])
            )

        if state.get("rag_results"):
            context_parts.append(
                "Knowledge base:\n" + "\n\n".join(state["rag_results"])
            )

        formatted_context = (
            "\n\n".join(context_parts)
            if context_parts
            else "None"
        )

        user_prompt = (
            f"Question:\n{state['question']}\n\n"
            f"Context:\n{formatted_context}"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state.get("history", []),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = str(response.content)

        except Exception as e:
            logger.error(f"Response LLM Error: {e}")
            answer = (
                "I couldn't generate a full reply due to a model error. "
                f"Here's what I found:\n{formatted_context}"
            )

        # Final output safety:
        # PII/secrets generated by the LLM, tools, MCP, memory or RAG
        # are removed before the answer is returned to the user.
        safe_answer, output_pii_detected = self.safety.sanitize_output(answer)

        if output_pii_detected:
            logger.warning("Agent Safety redacted PII from model output.")

        # Only safe/redacted input and safe output are stored in memory.
        self.memory.add("user", state["question"])
        self.memory.add("assistant", safe_answer)

        return {
            "answer": safe_answer,
            "pii_detected": (
                state.get("pii_detected", False) or output_pii_detected
            ),
        }

    def invoke(self, question: str) -> str:
        cleaned = question.strip() if question else ""

        if not cleaned:
            raise ValueError("Question cannot be empty.")

        result = self.graph.invoke({"question": cleaned})
        return result["answer"]


if __name__ == "__main__":
    agent = Orchestrator()

    print("AI Weather, Clothing & Calendar Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            inp = input("You: ").strip()

            if inp.lower() in {"exit", "quit"}:
                break

            if inp:
                print(f"Assistant: {agent.invoke(inp)}\n")

        except KeyboardInterrupt:
            break

        except Exception as exc:
            logger.error(f"Error: {exc}")
