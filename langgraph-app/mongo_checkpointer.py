from __future__ import annotations

import os

from langgraph.checkpoint.mongodb import MongoDBSaver


def _mongodb_uri() -> str:
    uri = os.getenv("LS_MONGODB_URI") or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "Mongo checkpointing requires LS_MONGODB_URI or MONGODB_URI. "
            "Use a replica-set-capable URI, for example "
            "mongodb://mongodb:27017/langgraph?replicaSet=rs0."
        )

    if not uri.startswith("mongodb+srv://") and "replicaSet=" not in uri:
        print(
            "WARNING: Mongo checkpointing should use a replica-set-capable URI. "
            "Set LS_MONGODB_URI with ?replicaSet=rs0 for Docker dev."
        )
    return uri


def make_checkpointer():
    """LangGraph custom checkpointer entrypoint backed by MongoDB."""

    return MongoDBSaver.from_conn_string(
        conn_string=_mongodb_uri(),
        db_name=os.getenv("ALPHARAVIS_CHECKPOINT_DB", "langgraph_checkpoints"),
        checkpoint_collection_name=os.getenv(
            "ALPHARAVIS_CHECKPOINT_COLLECTION",
            "checkpoints",
        ),
        writes_collection_name=os.getenv(
            "ALPHARAVIS_CHECKPOINT_WRITES_COLLECTION",
            "checkpoint_writes",
        ),
    )
