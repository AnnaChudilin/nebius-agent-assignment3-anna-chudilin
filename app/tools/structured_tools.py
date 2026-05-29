# Structured Tools for Customer Service Dataset Analysis
# These tools are designed to be invoked by the agent's reasoning process to perform specific data analysis tasks on the Bitext customer service dataset.
# Each tool is decorated with @tool to enable seamless integration with the agent's tool-calling mechanism, allowing for dynamic execution based on user queries.

from __future__ import annotations

import json
import pandas as pd
from langchain_core.tools import tool
from app.dataset import get_active_session, get_dataset_path

from app.profile import UserProfile


def _json_response(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _load_dataset(path: str | None = None) -> pd.DataFrame:
    dataset_path = get_dataset_path() or path or "data/bitext.csv"
    return pd.read_csv(dataset_path)


def _find_first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    candidates_lower = {candidate.lower().strip() for candidate in candidates}
    for col in df.columns:
        normalized = col.lower().strip()
        if normalized in candidates_lower or any(candidate in normalized for candidate in candidates_lower):
            return col
    return None


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def _resolve_target_column(df: pd.DataFrame, candidates: list[str], fallback: str) -> str | None:
    return _find_first_matching_column(df, candidates) or fallback if fallback in df.columns else None


def _build_examples(records: list[dict]) -> str:
    return _json_response({"examples": records})


@tool
def intents_by_category(category: str) -> str:
    """Return intent distribution for a given category."""
    try:
        df = _load_dataset()
        cat_col = _find_first_matching_column(df, ["category", "cat", "group"])
        intent_col = _find_first_matching_column(
            df, ["intent", "label", "intent_name"])

        if not cat_col or not intent_col:
            return _json_response({"error": "Could not find category or intent columns in the dataset."})

        target_cat = category.lower().strip()
        filtered_df = df[_normalize_series(df[cat_col]) == target_cat]
        if filtered_df.empty:
            return _json_response({"error": f"No data found for category '{category}'."})

        distribution = filtered_df[intent_col].value_counts().to_dict()
        return _json_response({"category": category, "distribution": distribution})
    except Exception as error:
        return _json_response({"error": f"Pipeline failed: {error}"})


@tool
def count_by_intent(intent: str) -> str:
    """Count dataset records for a specific intent."""
    try:
        df = _load_dataset()
        target_col = _find_first_matching_column(
            df, ["intent", "label", "intent_name"])
        if target_col is None:
            return _json_response({"error": "Could not find an intent column in the CSV."})

        clean_query = intent.lower().strip()
        csv_series = _normalize_series(df[target_col])

        if clean_query in ["refund", "refunds"]:
            filtered_df = df[csv_series == "get_refund"]
        elif clean_query in ["complaint", "complaints"]:
            filtered_df = df[csv_series == "complaint"]
        else:
            filtered_df = df[csv_series == clean_query]

        return _json_response({"count": len(filtered_df)})
    except Exception as error:
        return _json_response({"error": f"Error executing data evaluation loop: {error}"})


@tool
def list_categories() -> str:
    """List all unique dataset categories."""
    try:
        df = _load_dataset()
        target_col = _find_first_matching_column(
            df, ["category", "cat", "group", "label"])
        if target_col is None:
            return _json_response({"error": "Could not find a category column."})

        categories = sorted(df[target_col].dropna().astype(
            str).str.upper().unique().tolist())
        return _json_response({"categories": categories})
    except Exception as error:
        return _json_response({"error": f"Error retrieving text categories: {error}"})


@tool
def list_intents() -> str:
    """List all distinct intents in the dataset."""
    try:
        df = _load_dataset()
        target_col = _find_first_matching_column(
            df, ["intent", "label", "intent_name"])
        if target_col is None:
            return _json_response({"error": "Could not find an intent column."})

        intents = sorted(df[target_col].dropna().astype(
            str).str.lower().unique().tolist())
        return _json_response({"intents": intents})
    except Exception as error:
        return _json_response({"error": f"Error retrieving customer intents: {error}"})


@tool
def list_examples(category: str | None = None, count: int | None = 3, limit: int | None = None, session_id: str | None = "default") -> str:
    """Return sample dataset rows for a category with pagination support."""
    try:
        from app.profile import UserProfile

        df = _load_dataset()
        session_id = get_active_session()
        profile = UserProfile(session_id or "default")

        target_col = _find_first_matching_column(
            df, ["category", "cat", "group", "label"]) or "category"
        active_cat = category or profile.get_pagination_state().get(
            "last_category", "default")
        if not active_cat or active_cat == "default":
            active_cat = str(df[target_col].iloc[0]).lower().strip()

        filtered_df = df[_normalize_series(
            df[target_col]) == active_cat.lower().strip()]
        if filtered_df.empty:
            return _json_response({"examples": [], "message": f"No examples found for category '{active_cat}'"})

        final_size = limit if limit is not None else count
        if final_size is None or final_size <= 0:
            final_size = 3

        pag_state = profile.get_pagination_state()
        current_offset = pag_state.get("offset", 0)
        if pag_state.get("last_category") != active_cat.lower().strip() or current_offset >= len(filtered_df):
            current_offset = 0

        sliced_df = filtered_df.iloc[current_offset: current_offset + final_size]
        profile.set_pagination_state(
            active_cat, current_offset + len(sliced_df))

        if sliced_df.empty:
            sliced_df = filtered_df.head(final_size)
            profile.set_pagination_state(active_cat, len(sliced_df))

        display_cols = [c for c in ["instruction", "response",
                                    "utterance", "text"] if c in sliced_df.columns]
        if not display_cols:
            display_cols = list(sliced_df.columns)

        return _build_examples(sliced_df[display_cols].to_dict(orient="records"))
    except Exception as error:
        return _json_response({"error": f"Error retrieving sample layout examples: {error}"})


@tool
def intent_distribution() -> str:
    """Return intent distribution statistics for the dataset."""
    try:
        df = _load_dataset()
        target_col = _find_first_matching_column(
            df, ["intent", "label", "intent_name"])
        if target_col is None:
            return _json_response({"error": "Could not find an intent column."})

        raw_distribution = df[target_col].value_counts().to_dict()
        distribution = {str(k).lower(): v for k, v in raw_distribution.items()}
        return _json_response({"intent_distribution": distribution})
    except Exception as error:
        return _json_response({"error": f"Error evaluating intent balancing metrics: {error}"})
