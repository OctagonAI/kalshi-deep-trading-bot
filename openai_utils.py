"""
Utilities for working with OpenAI Responses API (GPT-5 compatible).

This module provides helpers to:
- Create responses with message-style input
- Extract the completed assistant message (handles reasoning-first outputs)
- Parse structured outputs into Pydantic models using Responses API
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def _normalize_messages_input(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert chat-style messages into Responses API input format (list of role/content dicts).

    Accepts already-correct structures and passes them through unchanged.
    """
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        # If content is already a list of parts, keep as-is; otherwise wrap text
        content = msg.get("content")
        if isinstance(content, list):
            normalized.append({"role": msg.get("role", "user"), "content": content})
        else:
            normalized.append({"role": msg.get("role", "user"), "content": str(content) if content is not None else ""})
    return normalized


def extract_completed_message_text(response: Any) -> str:
    """
    Extract plain text from the completed assistant message in a Responses API result.

    Handles outputs that include a reasoning step before the completed message.
    Returns an empty string if nothing is found.
    """
    text_chunks: List[str] = []

    try:
        output_items = getattr(response, "output", None) or response.get("output", [])  # type: ignore[attr-defined]
    except Exception:
        output_items = []

    for item in output_items or []:
        # SDK objects expose attributes; JSON dicts expose keys – support both
        item_type = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
        item_status = getattr(item, "status", None) or (isinstance(item, dict) and item.get("status"))
        if item_type == "message" and item_status == "completed":
            content = getattr(item, "content", None) or (isinstance(item, dict) and item.get("content")) or []
            for part in content or []:
                part_type = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                if part_type == "output_text":
                    # SDK: part.text; JSON: part["text"]
                    text_value = getattr(part, "text", None) or (isinstance(part, dict) and part.get("text"))
                    if isinstance(text_value, str) and text_value:
                        text_chunks.append(text_value)
            break

    return "".join(text_chunks).strip()


async def responses_create_text(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    reasoning_effort: str = "minimal",
    text_verbosity: str = "medium",
) -> str:
    """
    Create a Responses API call for free-form text and extract the assistant text.

    Falls back to an empty string if no completed message is present.
    """
    normalized = _normalize_messages_input(messages)
    response = await client.responses.create(
        model=model,
        input=normalized,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": text_verbosity},
    )

    return extract_completed_message_text(response)


async def responses_parse_pydantic(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    response_format: Type[T],
    reasoning_effort: str = "minimal",
    text_verbosity: str = "medium",
) -> T:
    """
    Parse structured output using Responses API into the provided Pydantic model type.

    Supports SDK objects exposing `output_parsed` or dicts with the same key.
    As a fallback, will attempt to locate a completed message and read a `.parsed` field
    if present on that message (for compatibility with interim SDK behaviors).
    """
    normalized = _normalize_messages_input(messages)

    # Prefer native parse if available in SDK
    if hasattr(client, "responses") and hasattr(client.responses, "parse"):
        resp = await client.responses.parse(
            model=model,
            input=normalized,
            response_format=response_format,
            reasoning={"effort": reasoning_effort},
            text={"verbosity": text_verbosity},
        )

        parsed = getattr(resp, "output_parsed", None)
        if parsed is not None:
            return cast(T, parsed)

        # Fallback: try to find parsed on the completed message item
        try:
            output_items = getattr(resp, "output", None) or resp.get("output", [])  # type: ignore[attr-defined]
        except Exception:
            output_items = []
        for item in output_items or []:
            item_type = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            item_status = getattr(item, "status", None) or (isinstance(item, dict) and item.get("status"))
            if item_type == "message" and item_status == "completed":
                candidate = getattr(item, "parsed", None) or (isinstance(item, dict) and item.get("parsed"))
                if candidate is not None:
                    return cast(T, candidate)

    # Absolute fallback: raise a helpful error
    raise RuntimeError("Structured output parsing failed: no output_parsed found in Responses API result.")


