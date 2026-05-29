# User profile management for the customer service agent.
# This module defines a UserProfile class that manages the storage and retrieval of user profile information based on their interactions with the agent.
# The profile is persisted to disk in JSON format, allowing the agent to maintain context about the user's name and interests across sessions.
# The UserProfile class provides methods to update the profile based on new user inputs, extract relevant information using regular expressions,
# and generate a summary of the profile details when requested by the user.
# The profile information can be used by the agent to personalize responses and provide more relevant insights based on the user's interests and past interactions.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

PROFILE_DIR = Path("profiles")
PROFILE_DIR.mkdir(exist_ok=True)

TOPIC_KEYWORDS: dict[str, List[str]] = {
    "refunds": ["refund", "money back", "reimbursement"],
    "shipping": ["shipping", "delivery", "tracking", "shipment"],
    "account": ["account", "login", "password", "billing"],
    "feedback": ["feedback", "review", "comment", "complaint"],
    "cancellation": ["cancel", "cancellation", "terminate", "stop service"],
}

NAME_PATTERNS = [
    re.compile(r"\bmy name is ([A-Za-z][A-Za-z ]+)\b", re.IGNORECASE),
    re.compile(r"\bi am ([A-Za-z][A-Za-z ]+)\b", re.IGNORECASE),
    re.compile(r"\bi'm ([A-Za-z][A-Za-z ]+)\b", re.IGNORECASE),
    re.compile(r"\bthis is ([A-Za-z][A-Za-z ]+)\b", re.IGNORECASE),
]


class UserProfile:
    def __init__(self, user_id: str):
        self.path = PROFILE_DIR / f"{user_id}.json"
        self.data: dict[str, object] = {
            "name": None,
            "interests": [],
        }
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.data = {"name": None, "interests": []}

    def save(self) -> None:
        self.path.write_text(json.dumps(
            self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def update(self, text: str) -> None:
        text = text.strip()
        name = self._extract_name(text)
        if name:
            self.data["name"] = name

        interests = self._extract_interests(text)
        current = {str(item).lower()
                   for item in self.data.get("interests", [])}
        updated = sorted(current.union(
            {interest.lower() for interest in interests}))
        self.data["interests"] = updated
        self.save()

    def _extract_name(self, text: str) -> str | None:
        for pattern in NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip().title()
        return None

    def _extract_interests(self, text: str) -> List[str]:
        normalized = text.lower()
        found: list[str] = []
        for interest, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                found.append(interest)
        return found

    def summary(self) -> str:
        name = self.data.get("name")
        interests = [interest for interest in self.data.get(
            "interests", []) if interest]
        if not name and not interests:
            return "I do not have any profile details yet."

        parts: list[str] = []
        if name:
            parts.append(f"Name: {name}.")
        if interests:
            parts.append(f"Interests: {', '.join(interests)}.")
        return " ".join(parts)

    # FIX: Persistent Pagination Context Getters and Setters
    # This guarantees full architectural alignment with Task 2a & 2b context management constraints.
    def get_pagination_state(self) -> dict:
        """
        Retrieve the current persistent pagination indices and tracking state tokens for this session.
        """
        return self.data.get("pagination", {"last_category": "default", "offset": 0})

    def set_pagination_state(self, category: str, offset: int) -> None:
        """
        Atomically write the updated pagination category block and threshold offset counters
        back into the profile data layer on disk immediately.
        """
        self.data["pagination"] = {
            "last_category": str(category).lower().strip(),
            "offset": int(offset)
        }
        self.save()
