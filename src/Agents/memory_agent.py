import json
import os


class MemoryAgent:

    def __init__(self, session_id="default", max_messages=10):
        self.session_id = session_id
        self.max_messages = max_messages
        self.file = "memory.json"
        self.messages = self._load()

    def _load(self):
        if not os.path.exists(self.file):
            return []

        try:
            with open(self.file, "r", encoding="utf-8") as file:
                data = json.load(file)

            return data.get(self.session_id, [])[-self.max_messages:]

        except (json.JSONDecodeError, OSError):
            return []

    def add(self, role, content):
        self.messages.append({
            "role": role,
            "content": content
        })

        self.messages = self.messages[-self.max_messages:]
        self._save()

    def get(self):
        return self.messages

    def _save(self):
        data = {}

        if os.path.exists(self.file):
            try:
                with open(self.file, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except (json.JSONDecodeError, OSError):
                data = {}

        data[self.session_id] = self.messages

        with open(self.file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

    def clear(self):
        self.messages = []
        self._save()