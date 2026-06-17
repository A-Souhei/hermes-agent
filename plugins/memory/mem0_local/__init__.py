"""mem0 self-hosted memory plugin — uses OSS mem0.Memory with local Qdrant.

No cloud account required. Requires:
  - Qdrant running locally (default: http://localhost:6333)
  - fastembed (local embeddings, no API key)
  - OPENAI_API_KEY env var (for mem0 fact extraction LLM)

Config via $HERMES_HOME/mem0_local.json (optional overrides):
  qdrant_host          — Qdrant host (default: localhost)
  qdrant_port          — Qdrant port (default: 6333)
  collection_name      — Qdrant collection (default: hermes_memories)
  embed_model          — fastembed model (default: BAAI/bge-small-en-v1.5)
  embedding_model_dims — vector dimensions for Qdrant collection (default: 384)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


def _load_config() -> dict:
    from hermes_constants import get_hermes_home
    cfg = {
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "collection_name": "hermes_memories",
        "embed_model": "BAAI/bge-small-en-v1.5",
        "embedding_model_dims": 384,
    }
    config_path = get_hermes_home() / "mem0_local.json"
    if config_path.exists():
        try:
            cfg.update(json.loads(config_path.read_text()))
        except Exception:
            pass
    return cfg


def _build_mem0_config(cfg: dict) -> dict:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": cfg["qdrant_host"],
                "port": cfg["qdrant_port"],
                "collection_name": cfg["collection_name"],
                "embedding_model_dims": cfg.get("embedding_model_dims", 384),
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {
                "model": cfg["embed_model"],
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "api_key": openai_key,
            },
        },
    }


SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Use when you need to recall something specific about the user or past context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Use for explicit preferences, "
        "corrections, or decisions the user wants remembered."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


class Mem0LocalMemoryProvider(MemoryProvider):
    """Self-hosted mem0 memory — local Qdrant + fastembed, no cloud account."""

    def __init__(self):
        self._memory: Optional[Any] = None
        self._lock = threading.Lock()
        self._user_id = "hermes-user"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0_local"

    def is_available(self) -> bool:
        try:
            import mem0  # noqa
            import fastembed  # noqa
            return bool(os.environ.get("OPENAI_API_KEY", ""))
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._user_id = kwargs.get("user_id") or "hermes-user"
        cfg = _load_config()
        mem0_cfg = _build_mem0_config(cfg)
        with self._lock:
            if self._memory is None:
                from mem0 import Memory
                self._memory = Memory.from_config(mem0_cfg)
        logger.info("[mem0_local] Initialized (user=%s, qdrant=%s:%s)",
                    self._user_id, cfg["qdrant_host"], cfg["qdrant_port"])

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS

    def system_prompt_block(self) -> str:
        return (
            "# Memory (mem0 local)\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to recall facts, mem0_conclude to store new ones."
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                with self._lock:
                    mem = self._memory
                results = mem.search(query, top_k=5, filters={"user_id": self._user_id})
                if isinstance(results, dict):
                    results = results.get("results", [])
                lines = [r.get("memory", "") for r in results if r.get("memory")]
                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("[mem0_local] prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0l-prefetch")
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return f"## Memory\n{result}" if result else ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _sync():
            try:
                with self._lock:
                    mem = self._memory
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                mem.add(messages, user_id=self._user_id)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("[mem0_local] sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0l-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({"error": "Memory temporarily unavailable."})

        with self._lock:
            mem = self._memory

        if tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = mem.search(query, top_k=top_k, filters={"user_id": self._user_id})
                if isinstance(results, dict):
                    results = results.get("results", [])
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                mem.add([{"role": "user", "content": conclusion}], user_id=self._user_id, infer=False)
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


def register(ctx) -> None:
    ctx.register_memory_provider(Mem0LocalMemoryProvider())
