# Summarization Tools for Customer Service Dataset
# This module defines tools that generate high-level summaries of customer interactions within specific categories of the Bitext dataset.
# These tools are designed to be called by the agent when a user requests an overview or summary of a category, providing insights into common intents, interaction patterns, and representative examples.

from __future__ import annotations

import json
import pandas as pd
from langchain_core.tools import tool
from app.dataset import get_dataset_path


def _json_response(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _load_dataset() -> pd.DataFrame:
    path = get_dataset_path() or "data/bitext.csv"
    return pd.read_csv(path)


def _find_column(df: pd.DataFrame, candidates: list[str], default: str) -> str:
    candidates_lower = {candidate.lower().strip() for candidate in candidates}
    for col in df.columns:
        normalized = col.lower().strip()
        if normalized in candidates_lower or any(candidate in normalized for candidate in candidates_lower):
            return col
    return default


def _normalize_text(value: str | pd.Series) -> str | pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(str).str.lower().str.strip()
    return value.lower().strip()


def _map_category_input(raw_category: str) -> str:
    mapping = {
        "cancellation": "cancel",
        "cancellations": "cancel",
        "refunds": "refund",
        "accounts": "account",
    }
    normalized = _normalize_text(raw_category)
    return mapping.get(normalized, normalized)


def _sample_texts(df: pd.DataFrame, text_columns: list[str], sample_size: int) -> list[str]:
    if not text_columns:
        return []
    column = text_columns[0]
    return df[column].dropna().astype(str).sample(n=sample_size, random_state=42).tolist()


def _build_summary(category: str, filtered_df: pd.DataFrame) -> str:
    total_records = len(filtered_df)
    text_columns = [c for c in ["response", "instruction",
                                "utterance", "text"] if c in filtered_df.columns]
    summary = f"Total records count for the aggregated {category.upper()} category: {total_records}. "
    if not text_columns:
        return summary + "No descriptive text columns were found to extract qualitative summaries."

    sample_size = min(15, total_records)
    sample_texts = _sample_texts(filtered_df, text_columns, sample_size)
    text_payload = " | ".join(sample_texts)
    return summary + f"Sample agent workflows and common textual phrases found in this context: {text_payload}"


@tool
def summarize_category(category: str) -> str:
    """Summarize dataset interactions for a given category or intent."""
    try:
        df = _load_dataset()
        category_column = _find_column(df, ["category", "label"], "category")
        intent_column = _find_column(df, ["intent"], "intent")

        clean_input = _map_category_input(category)
        filtered_df = df[_normalize_text(
            df[category_column].astype(str)) == clean_input]

        if filtered_df.empty and intent_column in df.columns:
            intent_matches = df[df[intent_column].astype(
                str).str.lower().str.contains(clean_input, na=False)]
            if not intent_matches.empty:
                detected_category = intent_matches[category_column].iloc[0]
                filtered_df = df[df[category_column] == detected_category]
                category = str(detected_category)

        if filtered_df.empty:
            return _json_response({"summary": f"No records found for the category or intent '{category}'."})

        return _json_response({"summary": _build_summary(category, filtered_df)})
    except Exception as error:
        return _json_response({"error": f"Error executing data evaluation loop: {error}"})
