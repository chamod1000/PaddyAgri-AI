"""
Text Sanitizer (core/synthesis/text_sanitizer.py)

Presentation-layer safety utilities shared by every farmer-facing render path.
Pure functions: no LLM calls, no I/O, no state.

Two guarantees:
  1. coerce_scalar()     — never emits a Python object repr into the response.
  2. summarize_snippet() — never emits a raw PDF chunk into the response.

Both are last-line-of-defence guards. A producer that supplies the wrong type
is still a bug in the producer; these functions make that bug visible on stdout
and harmless to the farmer, instead of invisible and user-facing.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, List

# Returned when a value cannot be safely rendered as prose.
UNRENDERABLE = ""

# PDF extraction artifacts that must never reach a farmer.
_PDF_ARTIFACTS = (
    r"\bPage\s+\d+\s*(?:of\s+\d+)?\b",
    r"\bFig(?:ure)?\.?\s*\d+[a-z]?\b",
    r"\bTable\s+\d+\b",
    r"[.·_—-]{4,}",
    r"\s+\d+\s*\|\s*",
)

# Operator-facing severity tags emitted by the deterministic rule engine
# (e.g. "[WARNING] ", "[RAIN ALERT] "). Internal telemetry, never farmer prose.
_LOG_TAG_PATTERN = r"^\s*\[[A-Z][A-Z0-9 _/-]{1,24}\]\s*"

# Abbreviations whose trailing period must not be treated as a sentence end.
_ABBREVIATIONS = (
    "Dr.", "Mr.", "Mrs.", "Ms.", "No.", "sp.", "spp.",
    "var.", "cv.", "approx.", "e.g.", "i.e.", "vs.",
)


def _warn(field: str, value: Any) -> None:
    """Report a producer contract violation without echoing the leaked repr."""
    label = field or "<unnamed>"
    logging.warning(
        f"[RENDER CONTRACT WARNING] Field '{label}' received unrenderable "
        f"{type(value).__name__}; omitted from response. "
        f"Producer must supply a string, number, or list of those."
    )


def coerce_scalar(value: Any, *, field: str = "", warn: bool = True) -> str:
    """
    Convert a value to farmer-safe display text, or return UNRENDERABLE ("").

    Renders : str, bool, int, float, Enum, and flat sequences of those.
    Refuses : pydantic models, dataclasses, dicts, and every other object.

    Refusing is the point. A pydantic model reaching this function used to be
    stringified into the response as
    "fungal_risk_alert='HIGH RISK' advisory_notes=[...]".
    """
    if value is None:
        return UNRENDERABLE
    if isinstance(value, str):
        return value.strip()
    # bool before int: bool is a subclass of int.
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Enum):
        return coerce_scalar(value.value, field=field, warn=warn)
    if isinstance(value, (list, tuple, set)):
        parts = [coerce_scalar(v, field=field, warn=warn) for v in value]
        return ", ".join(p for p in parts if p)
    if warn:
        _warn(field, value)
    return UNRENDERABLE


def strip_log_tags(text: str) -> str:
    """
    Remove internal severity tags from advisory prose.

    The rule engine emits notes prefixed "[WARNING] ", "[RAIN ALERT] " etc.
    Those are operator-facing log markers, not farmer-facing language.
    """
    cleaned = re.sub(_LOG_TAG_PATTERN, "", str(text))
    return " ".join(cleaned.split()).strip()


def _strip_pdf_artifacts(text: str) -> str:
    cleaned = " ".join(str(text).split())
    for pattern in _PDF_ARTIFACTS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip(" -–—|·.")


def _split_sentences(text: str) -> List[str]:
    protected = text
    for idx, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", f"<<{idx}>>"))

    out: List[str] = []
    for piece in re.split(r"(?<=[.!?])\s+", protected):
        for idx in range(len(_ABBREVIATIONS)):
            piece = piece.replace(f"<<{idx}>>", ".")
        piece = piece.strip()
        if piece:
            out.append(piece)
    return out


def summarize_snippet(raw: Any, *, max_sentences: int = 2, max_chars: int = 260) -> str:
    """
    Reduce a retrieved DOA chunk to a short, readable extract.

    Replaces the previous `chunk[:250] + "..."` behaviour, which cut mid-word
    and mid-sentence and read as a raw PDF dump.
    """
    text = coerce_scalar(raw, warn=False)
    if not text:
        return UNRENDERABLE

    text = _strip_pdf_artifacts(text)
    if not text:
        return UNRENDERABLE

    sentences = _split_sentences(text)
    if not sentences:
        return UNRENDERABLE

    summary = ""
    for sentence in sentences[:max_sentences]:
        candidate = f"{summary} {sentence}".strip() if summary else sentence
        if summary and len(candidate) > max_chars:
            break
        summary = candidate

    if len(summary) > max_chars:
        clipped = summary[:max_chars]
        boundary = clipped.rfind(" ")
        summary = (clipped[:boundary] if boundary > max_chars // 2 else clipped).rstrip(" ,;:")

    if summary and summary[-1] not in ".!?":
        summary += "."
    return summary
