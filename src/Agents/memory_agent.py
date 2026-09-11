from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class MemoryAgent:
    """Manages short-term conversation history."""

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")

        self.max_messages = max_messages
        self._messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        self._messages = self._messages[-self.max_messages :]

    def get(self) -> list[dict[str, str]]:
        return list(self._messages)

    def get_messages(self) -> list[BaseMessage]:
        messages: list[BaseMessage] = []

        for message in self._messages:
            if message["role"] == "user":
                messages.append(
                    HumanMessage(content=message["content"])
                )

            elif message["role"] == "assistant":
                messages.append(
                    AIMessage(content=message["content"])
                )

        return messages

    def clear(self) -> None:
        self._messages.clear()