# Agent construction using StateGraph ReAct workflow with custom prompt-based tool calling.
# This module defines a custom agent that interacts with the Bitext customer service dataset through a series of reasoning steps.
# The agent uses a structured prompt to determine when to call specific tools for data analysis,
# and it processes the tool outputs to generate final responses to user queries.
# The agent is built using the StateGraph framework to allow for flexible orchestration of LLM calls and tool executions,
# while maintaining a clear separation of concerns between reasoning and action.

from __future__ import annotations

import os
import warnings

from langchain_openai import ChatOpenAI
from app.checkpoint import SqliteCheckpointSaver
from app.dataset import set_dataset
from app.graph import compile_react_workflow, DEFAULT_SYSTEM_PROMPT
from app.tools.structured_tools import (
    intents_by_category,
    count_by_intent,
    intent_distribution,
    list_categories,
    list_intents,
    list_examples,
)
from app.tools.summarization_tools import summarize_category
from app.tools.utility_tools import calculate_expression

DEFAULT_API_BASE = "https://nebius.com"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _configure_nebius() -> tuple[str, str, str]:
    """Validates and configures environment variables for Nebius API access."""
    api_key = os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set NEBIUS_API_KEY (or OPENAI_API_KEY) before running the agent.")
    os.environ.setdefault("NEBIUS_API_KEY", api_key)

    base_url = os.getenv("NEBIUS_API_BASE") or os.getenv(
        "OPENAI_API_BASE") or DEFAULT_API_BASE
    if "://nebius.com" in base_url:
        warnings.warn(
            "NEBIUS_API_BASE points to ://nebius.com, which rejects tool-calling "
            f"requests with a chat-template error. Using {DEFAULT_API_BASE} instead.",
            stacklevel=2,
        )
        base_url = DEFAULT_API_BASE
    os.environ["NEBIUS_API_BASE"] = base_url

    model = os.getenv("NEBIUS_MODEL", DEFAULT_MODEL)
    return api_key, base_url, model


def build_agent(dataset_path: str | None = None):
    """Initializes the LLM and delegates state graph compilation to the workflow module."""
    api_key, base_url, model = _configure_nebius()
    set_dataset(dataset_path)

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_retries=2,
    )

    tools = [
        intents_by_category,
        list_categories,
        list_intents,
        count_by_intent,
        intent_distribution,
        list_examples,
        summarize_category,
        calculate_expression,
    ]

    # Delegate graph structure compiling to the specialized module
    return compile_react_workflow(
        llm=llm,
        tools=tools,
        checkpointer=SqliteCheckpointSaver()
    )
