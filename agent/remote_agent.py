"""Thin remote-agent client that delegates all agent work to a remote Hermes API server.

This module provides a drop-in replacement for `AIAgent` that sends messages to a
remote Hermes API server via its `/v1/chat/completions` endpoint and streams the
response back.  The local process needs no API keys, model config, or tool
registration — everything runs on the remote server.

The remote API server is the same endpoint served by `gateway/platforms/api_server.py`.
"""

import json
import logging
import urllib.parse
import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_HTTPX_TIMEOUT = 60 * 30  # 30 minutes — matches gateway proxy pattern


class RemoteAgent:
    """Delegates agent turns to a remote Hermes API server.

    Args:
        remote_url: Full URL of the remote API server (e.g. ``http://localhost:9119``).
            Must use ``http`` or ``https`` scheme.
        api_key: Optional Bearer token sent in the ``Authorization`` header.
        session_id: Session identifier sent via ``X-Hermes-Session-Id``.  Auto-generated
            UUID if ``None``.
        tool_progress_callback: Called on ``hermes.tool.progress`` SSE events with
            ``(event_type, function_name, preview, function_args, **kwargs)``.
        stream_delta_callback: Called for every content delta chunk during streaming.
        thinking_callback: Placeholder — not used in v1 chat-completions mode.
        clarify_callback: Placeholder — the v1 endpoint does not support mid-turn
            clarify prompts.
        print_fn: Drop-in for ``print()`` (e.g. Rich-compatible output).
        quiet_mode: Suppress internal log / print output.
    """

    _remote_url: str
    _api_key: Optional[str]
    _session_id: str
    _tool_progress_callback: Optional[Callable[..., Any]]
    _stream_delta_callback: Optional[Callable[[str], Any]]
    _thinking_callback: Optional[Callable[..., Any]]
    _clarify_callback: Optional[Callable[..., Any]]
    _print_fn: Optional[Callable[[str], Any]]
    _quiet_mode: bool
    _interrupted: bool

    # ------------------------------------------------------------------
    def __init__(
        self,
        remote_url: Optional[str] = None,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
        tool_progress_callback: Optional[Callable[..., Any]] = None,
        stream_delta_callback: Optional[Callable[[str], Any]] = None,
        thinking_callback: Optional[Callable[..., Any]] = None,
        clarify_callback: Optional[Callable[..., Any]] = None,
        print_fn: Optional[Callable[[str], Any]] = None,
        quiet_mode: bool = False,
    ) -> None:
        url = remote_url or "http://localhost:9119"
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"remote_url must use http or https scheme, got: {url!r}"
            )
        self._remote_url = url
        self._api_key = api_key
        self._session_id = session_id or uuid.uuid4().hex
        self._tool_progress_callback = tool_progress_callback
        self._stream_delta_callback = stream_delta_callback
        self._thinking_callback = thinking_callback
        self._clarify_callback = clarify_callback
        self._print_fn = print_fn
        self._quiet_mode = quiet_mode
        self._interrupted = False
        self._client: Optional[httpx.Client] = httpx.Client(timeout=_HTTPX_TIMEOUT)

    @property
    def session_id(self) -> str:
        """Public accessor for session ID — matches AIAgent interface."""
        return self._session_id

    def interrupt(self) -> None:
        """Abort any in-flight stream by closing the underlying HTTP client.

        Sets ``_interrupted`` so the exception handler in ``run_conversation``
        can distinguish a user-initiated abort from a real network error.
        """
        self._interrupted = True
        if self._client is not None:
            self._client.close()
            self._client = None

    def close(self) -> None:
        """Release the underlying HTTP client.  Safe to call multiple times."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    def chat(self, message: str, stream_callback: Optional[Callable[[str], Any]] = None) -> str:
        """Simple interface — send a message and return the final response string.

        Args:
            message: The user message to send.
            stream_callback: Called with each content delta as it arrives.

        Returns:
            The complete assistant response text.
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result.get("final_response", "")

    # ------------------------------------------------------------------
    def run_conversation(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
        stream_callback: Optional[Callable[[str], Any]] = None,
        persist_user_message: Optional[str] = None,  # not forwarded; server manages persistence
    ) -> Dict[str, Any]:
        """Full interface — send messages and stream the response.

        Matches the return shape of ``AIAgent.run_conversation()``:
        ``{"final_response": str, "messages": list, "api_calls": int, "tools": list, ...}``

        Args:
            user_message: The user's message.
            system_message: Optional system prompt.
            conversation_history: Optional list of messages in OpenAI format.
            task_id: Ignored by the remote agent (kept for interface compatibility).
            stream_callback: Called with each content delta as it arrives.

        Returns:
            A dict with ``final_response``, ``messages``, ``api_calls``, and ``tools``.
        """
        self._interrupted = False

        # Re-create client if it was closed by a previous interrupt()
        if self._client is None:
            self._client = httpx.Client(timeout=_HTTPX_TIMEOUT)

        # -- build messages ---------------------------------------------------
        messages: List[Dict[str, Any]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        # -- request payload --------------------------------------------------
        payload: Dict[str, Any] = {
            "model": "remote",
            "messages": messages,
            "stream": True,
        }

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers["X-Hermes-Session-Id"] = self._session_id

        full_response = ""

        # -- send request & stream --------------------------------------------
        try:
            with self._client.stream(
                "POST",
                f"{self._remote_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    try:
                        error_body = response.read().decode(errors="replace")
                        try:
                            error_detail = json.loads(error_body).get("error", {}).get("message", error_body[:200])
                        except json.JSONDecodeError:
                            error_detail = error_body[:200]
                    except Exception:
                        error_detail = "(no body)"
                    logger.warning(
                        "Remote agent returned HTTP %d from %s: %s",
                        response.status_code, self._remote_url, error_detail,
                    )
                    return {
                        "final_response": (
                            f"Remote agent returned HTTP {response.status_code} "
                            f"from {self._remote_url}"
                        ),
                        "messages": [],
                        "api_calls": 0,
                        "tools": [],
                    }

                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    logger.warning(
                        "Remote agent returned non-SSE content-type: %s",
                        content_type,
                    )
                    return {
                        "final_response": (
                            f"Remote agent returned unexpected content-type '{content_type}' "
                            f"from {self._remote_url}"
                        ),
                        "messages": [],
                        "api_calls": 0,
                        "tools": [],
                    }

                current_event: Optional[str] = None

                for line in response.iter_lines():
                    # -- blank line -------------------------------------------
                    if not line:
                        current_event = None
                        continue

                    # -- keepalive comment ------------------------------------
                    if line.startswith(":"):
                        continue

                    # -- event: line ------------------------------------------
                    if line.startswith("event: "):
                        current_event = line[len("event: "):].strip()
                        continue

                    # -- data: line -------------------------------------------
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]

                        # Stream end marker
                        if data_str == "[DONE]":
                            current_event = None
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            if current_event == "hermes.tool.progress":
                                logger.debug("Skipping malformed tool progress data")
                            current_event = None
                            continue

                        # -- tool progress event ------------------------------
                        if current_event == "hermes.tool.progress":
                            _handle_tool_progress(
                                data,
                                self._tool_progress_callback,
                                self._stream_delta_callback,
                            )
                            current_event = None
                            continue

                        # -- normal chat completion delta ---------------------
                        current_event = None  # reset after consuming

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        # Skip role-only deltas (e.g. {"role": "assistant"})
                        if "content" not in delta:
                            continue

                        content = delta.get("content")
                        if not content:
                            continue

                        full_response += content

                        if self._stream_delta_callback:
                            self._stream_delta_callback(content)
                        if stream_callback:
                            stream_callback(content)

        except httpx.ConnectError as exc:
            logger.warning("Failed to connect to remote agent at %s: %s", self._remote_url, exc)
            return {
                "final_response": f"Error connecting to remote agent at {self._remote_url}",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }
        except httpx.TimeoutException as exc:
            logger.warning("Remote agent request timed out at %s: %s", self._remote_url, exc)
            if full_response:
                result_messages = list(messages)
                result_messages.append({"role": "assistant", "content": full_response})
                return {
                    "final_response": full_response,
                    "messages": result_messages,
                    "api_calls": 1,
                    "tools": [],
                }
            return {
                "final_response": f"Remote agent request timed out at {self._remote_url}",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }
        except (httpx.ReadError, httpx.RemoteProtocolError) as exc:
            if self._interrupted:
                # User-initiated abort — return whatever was accumulated so far
                result_messages = list(messages)
                if full_response:
                    result_messages.append({"role": "assistant", "content": full_response})
                return {
                    "final_response": full_response,
                    "messages": result_messages,
                    "api_calls": 1 if full_response else 0,
                    "tools": [],
                }
            logger.warning("Remote agent stream error at %s: %s", self._remote_url, exc)
            if full_response:
                result_messages = list(messages)
                result_messages.append({"role": "assistant", "content": full_response})
                return {
                    "final_response": full_response,
                    "messages": result_messages,
                    "api_calls": 1,
                    "tools": [],
                }
            return {
                "final_response": f"Error connecting to remote agent at {self._remote_url}",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }
        except Exception as exc:
            if self._interrupted:
                result_messages = list(messages)
                if full_response:
                    result_messages.append({"role": "assistant", "content": full_response})
                return {
                    "final_response": full_response,
                    "messages": result_messages,
                    "api_calls": 1 if full_response else 0,
                    "tools": [],
                }
            logger.exception("Unexpected error during remote agent streaming: %s", exc)
            if full_response:
                result_messages = list(messages)
                result_messages.append({"role": "assistant", "content": full_response})
                return {
                    "final_response": full_response,
                    "messages": result_messages,
                    "api_calls": 1,
                    "tools": [],
                }
            return {
                "final_response": f"Error connecting to remote agent at {self._remote_url}",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        # -- success ----------------------------------------------------------
        # Preserve full conversation context — include system message and
        # history so the CLI can maintain multi-turn continuity.
        result_messages = list(messages)  # includes system + history + user
        result_messages.append({"role": "assistant", "content": full_response})
        return {
            "final_response": full_response,
            "messages": result_messages,
            "api_calls": 1,
            "tools": [],
        }


# ==========================================================================
# Helpers
# ==========================================================================

def _handle_tool_progress(
    data: Dict[str, Any],
    tool_progress_callback: Optional[Callable[..., Any]],
    stream_delta_callback: Optional[Callable[[str], Any]],
) -> None:
    """Parse a ``hermes.tool.progress`` SSE event and invoke callbacks."""
    tool_name = data.get("tool", "")
    status = data.get("status", "")
    emoji = data.get("emoji", "")
    label = data.get("label", "")
    function_args = data.get("args", "")

    if not tool_progress_callback:
        return

    if status == "running":
        tool_progress_callback(
            "start",
            tool_name,
            label or None,
            function_args or None,
        )
        if stream_delta_callback:
            stream_delta_callback(f"\n  {emoji} {label}...\n")
    elif status == "completed":
        tool_progress_callback(
            "complete",
            tool_name,
            None,
            None,
        )
