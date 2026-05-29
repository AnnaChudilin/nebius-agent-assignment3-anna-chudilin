# Checkpoint saver implementation using SQLite for persistence with thread safety.

from __future__ import annotations

import sqlite3
import pickle
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Iterator, Sequence, Optional

from langchain_core.runnables.base import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.base import ChannelVersions
from langgraph.checkpoint.serde.types import TASKS


class SqliteCheckpointSaver(BaseCheckpointSaver[str]):
    """Persist LangGraph checkpoints in a local SQLite database with thread-safe locks."""

    def __init__(self, path: str | Path = "checkpoints/checkpoints.db") -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Threading lock to prevent nested transactions across concurrent ReAct loops
        self.lock = threading.Lock()

        # Setting isolation_level=None disables Python's automatic transaction management.
        self.conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys = OFF")
        self._init_database()

    def _init_database(self) -> None:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        checkpoint BLOB NOT NULL,
                        metadata BLOB NOT NULL,
                        parent_checkpoint_id TEXT,
                        PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id)
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        write_idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        value BLOB NOT NULL,
                        task_path TEXT,
                        PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx)
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS blobs (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        version TEXT NOT NULL,
                        value BLOB NOT NULL,
                        PRIMARY KEY(thread_id, checkpoint_ns, channel, version)
                    )
                    """
                )
                self.conn.execute("COMMIT")
            except Exception as e:
                self.conn.execute("ROLLBACK")
                raise e

    def _blob_key(self, channel: str, version: Any) -> str:
        return str(version)

    def _load_pending_sends(
        self,
        thread_id: str,
        checkpoint_ns: str,
        parent_checkpoint_id: str | None,
    ) -> list[Any]:
        """Rebuild pending Send packets from the parent checkpoint's task writes."""
        if not parent_checkpoint_id:
            return []
        rows = self.conn.execute(
            "SELECT task_id, write_idx, value, task_path FROM writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? AND channel = ? "
            "ORDER BY task_path, task_id, write_idx",
            (thread_id, checkpoint_ns, parent_checkpoint_id, TASKS),
        ).fetchall()
        return [self.serde.loads_typed(pickle.loads(value)) for _, _, value, _ in rows]

    def _assemble_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_data: dict[str, Any],
        parent_checkpoint_id: str | None,
    ) -> dict[str, Any]:
        return {
            **checkpoint_data,
            "channel_values": self._load_blobs(
                thread_id, checkpoint_ns, checkpoint_data["channel_versions"]
            ),
            "pending_sends": self._load_pending_sends(
                thread_id, checkpoint_ns, parent_checkpoint_id
            ),
        }

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        """Load channel blobs from the database and deserialize them."""
        channel_values: dict[str, Any] = {}
        for channel, version in versions.items():
            row = self.conn.execute(
                "SELECT value FROM blobs WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?",
                (thread_id, checkpoint_ns, channel,
                 self._blob_key(channel, version)),
            ).fetchone()
            if not row:
                continue
            raw_value = row[0]
            if raw_value != b"empty":
                typed_data = pickle.loads(raw_value)
                channel_values[channel] = self.serde.loads_typed(typed_data)
        return channel_values

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a specific checkpoint tuple from the database with thread safety."""
        with self.lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id = get_checkpoint_id(config)

            query = (
                "SELECT checkpoint, metadata, parent_checkpoint_id "
                "FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? "
            )
            params = [thread_id, checkpoint_ns]

            if checkpoint_id:
                query += "AND checkpoint_id = ?"
                params.append(checkpoint_id)
            else:
                query += "ORDER BY checkpoint_id DESC LIMIT 1"

            row = self.conn.execute(query, params).fetchone()
            if not row:
                return None

            checkpoint_blob, metadata_blob, parent_checkpoint_id = row

            checkpoint: dict[str, Any] = self.serde.loads_typed(
                pickle.loads(checkpoint_blob))
            metadata = self.serde.loads_typed(pickle.loads(metadata_blob))

            if checkpoint_id is None:
                id_row = self.conn.execute(
                    "SELECT checkpoint_id FROM checkpoints "
                    "WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                ).fetchone()
                checkpoint_id = id_row[0] if id_row else None

            pending_writes = [
                (task_id, channel, self.serde.loads_typed(pickle.loads(value)))
                for task_id, _, channel, value in self.conn.execute(
                    "SELECT task_id, write_idx, channel, value FROM writes "
                    "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
                    "ORDER BY write_idx",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
            ]

            return CheckpointTuple(
                config=config,
                checkpoint=self._assemble_checkpoint(
                    thread_id, checkpoint_ns, checkpoint, parent_checkpoint_id
                ),
                metadata=metadata,
                pending_writes=pending_writes,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: dict[str, Any],
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Stores a conversation state snapshot inside the database under a thread lock."""
        with self.lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id = checkpoint["id"]
            parent_checkpoint_id = config["configurable"].get("checkpoint_id")

            # Extract internal structural variables safely
            checkpoint_copy = checkpoint.copy()
            channel_versions = checkpoint_copy.get("channel_versions", {})
            channel_values = checkpoint_copy.pop("channel_values", {})
            checkpoint_copy.pop("pending_sends", None)

            self.conn.execute("BEGIN")
            try:
                for channel, version in new_versions.items():
                    if channel in channel_values:
                        serialized_typed = self.serde.dumps_typed(
                            channel_values[channel])
                        blob = pickle.dumps(serialized_typed)
                    else:
                        blob = b"empty"

                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO blobs (thread_id, checkpoint_ns, channel, version, value)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (thread_id, checkpoint_ns, channel,
                         self._blob_key(channel, version), blob),
                    )

                checkpoint_blob = pickle.dumps(
                    self.serde.dumps_typed(checkpoint_copy))
                metadata_blob = pickle.dumps(self.serde.dumps_typed(metadata))

                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints 
                    (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata, parent_checkpoint_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id,
                     checkpoint_blob, metadata_blob, parent_checkpoint_id),
                )
                self.conn.execute("COMMIT")
            except Exception as e:
                self.conn.execute("ROLLBACK")
                raise e

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Saves intermediate task state mutations ensuring isolated atomic transactions."""
        with self.lock:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id = config["configurable"]["checkpoint_id"]

            self.conn.execute("BEGIN")
            try:
                for idx, (channel, value) in enumerate(writes):
                    serialized_typed = self.serde.dumps_typed(value)
                    blob = pickle.dumps(serialized_typed)

                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO writes 
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx, channel, value)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (thread_id, checkpoint_ns, checkpoint_id,
                         task_id, idx, channel, blob),
                    )
                self.conn.execute("COMMIT")
            except Exception as e:
                self.conn.execute("ROLLBACK")
                raise e

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Lists historical state snapshots matching tracking filter tokens safely."""
        if config is None:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with self.lock:
            query = (
                "SELECT checkpoint_id, checkpoint, metadata, parent_checkpoint_id "
                "FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? "
            )
            params = [thread_id, checkpoint_ns]

            if before:
                query += "AND checkpoint_id < ? "
                params.append(before["configurable"]["checkpoint_id"])

            query += "ORDER BY checkpoint_id DESC "
            if limit:
                query += f"LIMIT {limit}"

            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()

        for row in rows:
            c_id, c_blob, m_blob, p_id = row

            checkpoint_data = self.serde.loads_typed(pickle.loads(c_blob))
            metadata_data = self.serde.loads_typed(pickle.loads(m_blob))

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": c_id,
                    }
                },
                checkpoint=self._assemble_checkpoint(
                    thread_id, checkpoint_ns, checkpoint_data, p_id
                ),
                metadata=metadata_data,
                pending_writes=[],
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": p_id,
                        }
                    }
                    if p_id else None
                ),
            )
