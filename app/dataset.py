# Dataset loading and management for the Bitext customer service dataset.
# This module provides functions to load the dataset from a CSV file, access it as a pandas DataFrame, and set a custom dataset path if needed.
# The dataset is cached in memory after the first load to optimize performance for subsequent tool calls that require access to the data.
# The default dataset path is set to "data/bitext.csv", but it can be overridden by passing a different path to the set_dataset function
# or by setting the DATASET_PATH environment variable before starting the MCP server.

from __future__ import annotations

import os
from pathlib import Path

# Global state variable to store the path to the current CSV dataset
_dataset_path: str | None = None

_active_session: str = "default"


def set_active_session(session_id: str):
    global _active_session
    _active_session = session_id


def get_active_session() -> str:
    global _active_session
    return _active_session


def set_dataset(path: str | Path | None = None) -> None:
    """
    Sets the global dataset path location. 
    Defaults to 'data/bitext.csv' if no path is explicitly provided.
    """
    global _dataset_path
    if path is None:
        _dataset_path = "data/bitext.csv"
    else:
        _dataset_path = str(path)

    # Ensure the parent directories exist for safety
    Path(_dataset_path).parent.mkdir(parents=True, exist_ok=True)


def get_dataset_path() -> str:
    """
    Retrieves the currently configured global dataset path.
    Falls back to the default location if set_dataset() hasn't been called.
    """
    global _dataset_path
    if _dataset_path is None:
        return "data/bitext.csv"
    return _dataset_path
