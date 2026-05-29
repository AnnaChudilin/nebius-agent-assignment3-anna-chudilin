# Query classification for routing user queries to appropriate tools or handling out-of-scope questions.
# This module defines a simple keyword-based classifier to determine whether an incoming user query is structured (i.e., likely to require specific tool calls),
# unstructured (i.e., requesting summaries or insights without clear structured query patterns), or out of scope (i.e., unrelated to the customer service dataset).
# The classify_query function analyzes the query text against predefined keyword lists to assign a query type and provide a reason for the classification, which can be used for logging and debugging purposes in the main application flow.
# This routing layer helps ensure that the agent responds appropriately to different types of user inputs and maintains focus on the relevant dataset analysis tasks.

from __future__ import annotations

from pydantic import BaseModel
from typing import Literal

STRUCTURED_KEYWORDS = [
    "more",         # Forces 'Show me 3 more' to lock cleanly into the structured graph node workflow
    "count",
    "how many",
    "distribution",
    "show",
    "examples",
    "category",
    "categories",  # Added plural form to capture 'what categories exist' accurately
    "intent",
    "intents",     # Added plural form for intent queries
    "top",
    "percentage",
    "total",
    "number",
    "refund",
    "refunds",    # Added plural form
    "shipping",
    "billing",
    "cancellation",
    "account",
    "order",
    "feedback",
    "support",
    "customer",
]

UNSTRUCTURED_KEYWORDS = [
    "summarize",
    "summary",
    "describe",
    "explain",
    "insight",
    "what do you remember",
    "what should i query",
    "what about",
    "how do",
    "how are",
]

OUT_OF_SCOPE_KEYWORDS = [
    "football",
    "president",
    "poem",
    "crm",
    "weather",
    "champions league",
    "who is",
    "best software",
    "who won",
    "who are",
    "where is",
    "when is",
    "why is",
    "what is the capital",
    "what is the president",
]

DATASET_SIGNAL_KEYWORDS = [
    "bitext",
    "customer service",
    "customer",
    "support",
    "dataset",
    "refund",
    "shipping",
    "billing",
    "account",
    "order",
    "cancellation",
    "intent",
    "category",
    "feedback",
]


class RouteResult(BaseModel):
    query_type: Literal["structured", "unstructured", "out_of_scope"]
    reason: str


def classify_query(query: str) -> RouteResult:
    """Classify the incoming user query before tools are selected."""
    normalized = query.strip().lower()
    if not normalized:
        return RouteResult(
            query_type="out_of_scope",
            reason="The query is empty or not meaningful.",
        )

    # 1. High-priority explicit out-of-scope validation pass
    if any(keyword in normalized for keyword in OUT_OF_SCOPE_KEYWORDS):
        return RouteResult(
            query_type="out_of_scope",
            reason="Detected a question that is unrelated to the customer service dataset.",
        )

    # Elevate explicit UNSTRUCTURED command heuristics (summarize, explain) above structured entity keywords.
    # This ensures "Summarize the FEEDBACK category" is classified as unstructured, exactly as Task 1 requires.
    if any(keyword in normalized for keyword in UNSTRUCTURED_KEYWORDS):
        return RouteResult(
            query_type="unstructured",
            reason="Detected a request for summary, explanation, or a high-level overview.",
        )

    # 3. Check for concrete data-driven actions (count, distribution, categories)
    if any(keyword in normalized for keyword in STRUCTURED_KEYWORDS):
        return RouteResult(
            query_type="structured",
            reason="Detected a concrete dataset question with structured query keywords.",
        )

    # 4. Dataset signal matching fallback layout
    if any(keyword in normalized for keyword in DATASET_SIGNAL_KEYWORDS):
        return RouteResult(
            query_type="unstructured",
            reason="Detected dataset-related language without explicit structured phrasing; treating this as an unstructured dataset request.",
        )

    return RouteResult(
        query_type="out_of_scope",
        reason="No dataset keywords found, so the question is likely unrelated to the Bitext customer service dataset.",
    )
