# State management for the customer service agent.
# This module defines the AgentState TypedDict, which encapsulates the current state of the agent during a user session.
# The state includes the conversation history (messages), the type of query being processed (query_type), the user ID and session ID for context, and a list of reasoning steps taken by the agent.
# This structured state representation allows for consistent tracking of the agent's interactions and reasoning process across different components of the application, such as memory management, profile handling, and tool execution.

from typing import TypedDict, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: List[BaseMessage]
    query_type: str
    user_id: str
    session_id: str
    reasoning_steps: List[str]
