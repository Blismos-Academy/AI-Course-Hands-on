import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from agents.memory_agent import MemoryAgent
from agents.planner import Planner
from agents.tool_agent import ToolAgent, weather_tools
from rag.retrival import HybridRetriever


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LANGSMITH CONFIGURATION
# ============================================================

LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "weather-ai-agent",
)


# ============================================================
# AGENT STATE
# ============================================================

class AgentState(TypedDict, total=False):
    question: str
    history: list[BaseMessage]
    route: str
    tool_results: list[str]
    rag_results: list[str]
    answer: str


# ============================================================
# ORCHESTRATOR
# ============================================================

class Orchestrator:

    def __init__(self):

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        self.llm = ChatGroq(
            model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b",
            ),
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY"),
        )

        # ----------------------------------------------------
        # AGENTS
        # ----------------------------------------------------

        self.memory = MemoryAgent(
            max_messages=10
        )

        self.planner = Planner(
            self.llm
        )

        self.tools = ToolAgent(
            self.llm,
            weather_tools
        )

        self.retriever = HybridRetriever(
            threshold=0.5
        )

        # ----------------------------------------------------
        # BUILD GRAPH
        # ----------------------------------------------------

        self.graph = self._build_graph()

    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    def _build_graph(self):

        graph = StateGraph(AgentState)

        # ----------------------------------------------------
        # NODES
        # ----------------------------------------------------

        graph.add_node(
            "memory",
            self._memory
        )

        graph.add_node(
            "planner",
            self._planner
        )

        graph.add_node(
            "tools",
            self._tools
        )

        graph.add_node(
            "rag",
            self._rag
        )

        graph.add_node(
            "response",
            self._response
        )

        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        graph.add_edge(
            START,
            "memory"
        )

        graph.add_edge(
            "memory",
            "planner"
        )

        # ----------------------------------------------------
        # PLANNER ROUTING
        # ----------------------------------------------------

        graph.add_conditional_edges(
            "planner",
            lambda state: state["route"],
            {
                "memory": "response",
                "rag": "rag",
                "tools": "tools",
                "both": "tools",
            },
        )

        # ----------------------------------------------------
        # TOOL ROUTING
        # ----------------------------------------------------

        graph.add_conditional_edges(
            "tools",
            lambda state: (
                "rag"
                if state["route"] == "both"
                else "response"
            ),
            {
                "rag": "rag",
                "response": "response",
            },
        )

        # ----------------------------------------------------
        # RAG -> RESPONSE
        # ----------------------------------------------------

        graph.add_edge(
            "rag",
            "response"
        )

        # ----------------------------------------------------
        # RESPONSE -> END
        # ----------------------------------------------------

        graph.add_edge(
            "response",
            END
        )

        return graph.compile()

    # ========================================================
    # MEMORY NODE
    # ========================================================

    @traceable(
        name="Memory Node",
        run_type="chain"
    )
    def _memory(self, state):

        history = self.memory.get_messages()

        return {
            "history": history
        }

    # ========================================================
    # PLANNER NODE
    # ========================================================

    @traceable(
        name="Planner Node",
        run_type="chain"
    )
    def _planner(self, state):

        route = self.planner.invoke(
            state["question"],
            state.get("history", []),
        )

        return {
            "route": route
        }

    # ========================================================
    # TOOL NODE
    # ========================================================

    @traceable(
        name="Tools Node",
        run_type="chain"
    )
    def _tools(self, state):

        tool_results = self.tools.invoke(
            state["question"],
            state.get("history", []),
        )

        return {
            "tool_results": tool_results
        }

    # ========================================================
    # RAG NODE
    # ========================================================

    @traceable(
        name="RAG Node",
        run_type="retriever"
    )
    def _rag(self, state):

        query = state["question"]

        # ----------------------------------------------------
        # Add live tool information to RAG query
        # ----------------------------------------------------

        if state.get("tool_results"):

            query += (
                "\n\nLive information:\n"
                + "\n".join(
                    state["tool_results"]
                )
            )

        # ----------------------------------------------------
        # Retrieve knowledge
        # ----------------------------------------------------

        rag_results = self.retriever.search(
            query,
            top_k=3,
        ) or []

        return {
            "rag_results": rag_results
        }

    # ========================================================
    # RESPONSE NODE
    # ========================================================

    @traceable(
        name="Response Node",
        run_type="chain"
    )
    def _response(self, state):

        context = []

        # ----------------------------------------------------
        # TOOL CONTEXT
        # ----------------------------------------------------

        if state.get("tool_results"):

            context.append(
                "Live information:\n"
                + "\n".join(
                    state["tool_results"]
                )
            )

        # ----------------------------------------------------
        # RAG CONTEXT
        # ----------------------------------------------------

        if state.get("rag_results"):

            context.append(
                "Knowledge base:\n"
                + "\n\n".join(
                    state["rag_results"]
                )
            )

        # ----------------------------------------------------
        # FINAL LLM
        # ----------------------------------------------------

        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a helpful trip-planning "
                        "assistant. "

                        "Answer the user's request using "
                        "the provided conversation and "
                        "context. "

                        "Do not invent information "
                        "and do not mention internal "
                        "implementation. "

                        "Keep the answer simple, accurate "
                        "and concise."
                    )
                ),

                *state.get(
                    "history",
                    []
                ),

                HumanMessage(
                    content=(
                        f"Question:\n"
                        f"{state['question']}\n\n"

                        f"Context:\n"
                        f"{chr(10).join(context) or 'None'}"
                    )
                ),
            ]
        )

        answer = str(
            response.content
        )

        # ----------------------------------------------------
        # MEMORY UPDATE
        # ----------------------------------------------------

        self.memory.add(
            "user",
            state["question"]
        )

        self.memory.add(
            "assistant",
            answer
        )

        return {
            "answer": answer
        }

    # ========================================================
    # MAIN INVOKE
    # ========================================================

    @traceable(
        name="Weather AI Agent",
        run_type="chain"
    )
    def invoke(
        self,
        question: str
    ) -> str:

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        # ----------------------------------------------------
        # LangGraph execution
        # ----------------------------------------------------

        result = self.graph.invoke(
            {
                "question": question
            },
            config={
                "run_name": "Weather AI Agent",
                "tags": [
                    "weather-agent",
                    "langgraph",
                    "production",
                ],
                "metadata": {
                    "project": LANGSMITH_PROJECT,
                    "question": question,
                },
            },
        )

        return result["answer"]

    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question: str
    ) -> str:

        return self.invoke(question)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    agent = Orchestrator()

    print(
        "AI Q&A Assistant"
    )

    print(
        "Type 'exit' to stop.\n"
    )

    while True:

        try:

            question = input(
                "You: "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
            }:
                break

            if question:

                answer = agent.invoke(
                    question
                )

                print(
                    "Assistant:",
                    answer
                )

        except KeyboardInterrupt:

            print("\nExiting...")
            break

        except Exception as exc:

            print(
                f"Error: {exc}"
            )