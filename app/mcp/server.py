# MCP Server Implementation for Customer Service Agent Network

from __future__ import annotations

import os
from fastmcp import FastMCP

from app.dataset import set_dataset
from app.tools.structured_tools import (
    intents_by_category,
    count_by_intent,
    intent_distribution,
    list_categories,
    list_intents,
    list_examples,
)
from app.tools.summarization_tools import summarize_category

# Initialize FastMCP server instance for the customer service agent network
mcp = FastMCP("customer-service-agent")

# Ensure dataset path is bound globally before server handling loops start
DEFAULT_DATASET_PATH = os.getenv("DATASET_PATH", "data/bitext.csv")
set_dataset(DEFAULT_DATASET_PATH)


def _unwrap_tool(tool_item):
    """
    Extracts the underlying Python function from LangChain tool wrappers 
    to preserve signature types and docstrings during MCP exposure.
    """
    if hasattr(tool_item, "func"):
        return tool_item.func
    if hasattr(tool_item, "__wrapped__"):
        return tool_item.__wrapped__
    return tool_item


# Iterate and register every analytical tool inside the FastMCP container
for tool in [
    list_categories,
    list_intents,
    count_by_intent,
    intents_by_category,
    list_examples,
    intent_distribution,
    summarize_category,
]:
    # Extract the clean callable function object
    tool_fn = _unwrap_tool(tool)

    # Use the explicit add_tool method for dynamic registration instead of calling decorators
    mcp.add_tool(tool_fn)


if __name__ == "__main__":
    # Launch the native MCP transport interface loop
    mcp.run()
