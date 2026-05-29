# Conversation memory management for the customer service agent.
# This module defines a ConversationMemory class that handles the storage and retrieval of conversation history for each user session.
# The memory is persisted to disk in JSON format, allowing the agent to maintain context across interactions.
# The ConversationMemory class provides methods to append new messages, load existing conversations, and clear memory when needed.
# Each session is identified by a unique session ID, which corresponds to a JSON file in the "sessions" directory.
# The memory structure is designed to be simple and efficient, enabling the agent to access past interactions and user inputs to inform its responses and reasoning processes.

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


class ConversationMemory:
    def __init__(self, session_id: str):
        self.path = SESSIONS_DIR / f"{session_id}.json"
        self.messages: List[dict[str, str]] = []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                raw = self.path.read_text(encoding="utf-8")
                self.messages = json.loads(raw)
                if not isinstance(self.messages, list):
                    self.messages = []
            except json.JSONDecodeError:
                self.messages = []
        else:
            self.messages = []

    def save(self) -> None:
        self.path.write_text(json.dumps(
            self.messages, indent=2, ensure_ascii=False), encoding="utf-8")

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.save()

    def to_messages(self) -> List[Tuple[str, str]]:
        return [(message["role"], message["content"]) for message in self.messages]

    def clear(self) -> None:
        self.messages = []
        self.save()
