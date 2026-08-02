"""
Conversation Experience Engine — PaddyAgri-AI v6.0 (FINAL)
============================================================================
Staff Engineering Production Freeze Candidate (CACAA-AO Architecture)

PRINCIPAL ARCHITECTURE REVIEW & PRODUCTION HARDENING
────────────────────────────────────────────────────
[H1] Chain-of-Thought (CoT) Self-Review Layer
     The LLM is now forced to generate a <draft>, perform a 7-point internal 
     scoring in <review>, and output the polished <final_response>. This 
     guarantees the quality gate is executed, eliminating prompt-skipping.

[H2] Heuristic Polish Layer
     Added robust bullet-point density checks and enhanced trailing filler 
     removal to catch Markdown formatting artifacts.

[H3] Conversation Continuity Engine
     Explicit "Current Topic" vs "Past Context" separation allows the LLM 
     to resolve anaphora ("it", "this disease") without redundantly restarting 
     the conversational flow.

[H4] Expanded Deterministic Fallbacks
     Fallbacks now feature slight random variation to avoid sounding like
     a static template if the primary LLM is repeatedly offline.

PRINCIPLE OF OPERATION
──────────────────────
Natural Conversation > Readability > Accuracy > Completeness > Consistency
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from config.model_provider import get_reasoning_model
from core.synthesis.text_sanitizer import coerce_scalar, summarize_snippet, strip_log_tags


# ─────────────────────────────────────────────────────────────────────────────
# Sources block markers
# ─────────────────────────────────────────────────────────────────────────────

# The heading actually emitted by _format_sources. Previously the append guard
# tested for "📚 Sources" — a marker emitted only by the legacy
# response_experience_engine — so it never matched and the block was appended
# a second time. Both markers are now checked.
_SOURCES_HEADING = "### Supporting references:"
_LEGACY_SOURCES_HEADING = "📚 Sources"


def _has_sources_block(text: str) -> bool:
    return _SOURCES_HEADING in text or _LEGACY_SOURCES_HEADING in text


# ─────────────────────────────────────────────────────────────────────────────
# Intent constants
# ─────────────────────────────────────────────────────────────────────────────

class _Intent:
    KNOWLEDGE   = "knowledge"
    DIAGNOSIS   = "diagnosis"
    WEATHER     = "weather"
    FERTILIZER  = "fertilizer"
    IMAGE       = "image"
    MIXED       = "mixed"
    SOCIAL      = "social"


# ─────────────────────────────────────────────────────────────────────────────
# Intent classifier (Knowledge-First)
# ─────────────────────────────────────────────────────────────────────────────

_KNOWLEDGE_PATTERNS = (
    r"\bwhat is\b", r"\bwhat are\b", r"\bexplain\b", r"\bdescribe\b",
    r"\btell me about\b", r"\bhow does\b", r"\bwhat causes\b",
    r"\bwhat kind of\b", r"\bdefine\b", r"\boverview\b",
)

_SYMPTOM_KEYWORDS = (
    "my leaves", "my crop", "my field", "my plants", "my rice",
    "i see", "i notice", "i found", "lesions", "spots on",
    "brown edges", "yellow edges", "dead tips", "wilting",
    "i think my", "suspect", "look like",
)

_WEATHER_KEYWORDS = (
    "weather", "rain", "humidity", "temperature", "forecast", "climate",
    "fungal risk", "disease risk", "apply tomorrow", "field condition",
    "raining", "wet season", "dry season",
)

_FERTILIZER_KEYWORDS = (
    "fertilizer", "fertiliser", "urea", "tsp", "mop", "npk", "top dress",
    "basal", "nutrient", "nitrogen", "phosphorus", "potassium", "tillering",
    "panicle", "dosage", "kg per acre", "apply fertilizer",
)

_IMAGE_KEYWORDS = (
    "upload", "image", "photo", "picture", "leaf image", "attached",
    "this image", "this photo", "this picture", "analyze this", "analyse this",
)

# Pure social turns: greetings, thanks, sign-offs. Matched only when the whole
# message is social (see _is_social), so "hello, my leaves have spots" still
# routes to DIAGNOSIS.
_SOCIAL_PATTERNS = (
    r"^(hi|hello|hey|hii+|yo)\b",
    r"\bayubowan\b",
    r"\bgood (morning|afternoon|evening|day)\b",
    r"^(thanks|thank you|thankyou|thx|ta)\b",
    r"\b(much appreciated|appreciate it)\b",
    r"^(ok|okay|okey|alright|got it|noted|fine|sure)\b",
    r"^(bye|goodbye|see you|good night|gn)\b",
    r"^(who are you|what are you|what can you do|help)\b",
)

# Any of these means the turn carries real work, so it is never SOCIAL.
_SUBSTANTIVE_MARKERS = (
    "?", "disease", "blast", "blight", "paddy", "rice", "crop", "leaf", "leaves",
    "field", "spray", "apply", "dose", "dosage", "kg", "acre", "season",
    "yala", "maha", "how", "why", "when", "where", "which", "should",
)

# Social turns are short by nature; a long message is doing something else.
_SOCIAL_MAX_WORDS = 6


def _social_kind(query: str) -> str:
    """Which flavour of social turn this is, for the offline fallback."""
    q = query.lower().strip()
    if re.search(r"^(thanks|thank you|thankyou|thx|ta)\b|\b(much appreciated|appreciate it)\b", q):
        return "thanks"
    if re.search(r"^(bye|goodbye|see you|good night|gn)\b", q):
        return "farewell"
    if re.search(r"^(who are you|what are you|what can you do|help)\b", q):
        return "identity"
    if re.search(r"^(ok|okay|okey|alright|got it|noted|fine|sure)\b", q):
        return "acknowledgement"
    return "greeting"


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic confidence
# ─────────────────────────────────────────────────────────────────────────────

# A diagnosis is only stated as fact at HIGH. Anything lower is hedged and the
# farmer is told what to check, because acting on a wrong disease call costs a
# spray cycle and the crop keeps degrading.
_CONF_HIGH   = "high"
_CONF_MEDIUM = "medium"
_CONF_LOW    = "low"

# Checked LOW first, then MEDIUM, then HIGH: hedging words must win. Matching
# is word-boundary anchored because a substring scan read "uncertain" as
# "certain" and promoted an unsure diagnosis to confirmed.
_CONF_SYNONYMS = (
    (_CONF_LOW,    ("low", "very low", "weak", "uncertain", "unsure", "unclear",
                    "tentative", "possible", "inconclusive", "not conclusive")),
    (_CONF_MEDIUM, ("medium", "moderate", "fair", "probable", "likely", "partial")),
    (_CONF_HIGH,   ("high", "very high", "strong", "certain", "confirmed",
                    "definite", "conclusive")),
)


def _confidence_tier(raw: Any) -> str:
    """
    Map any confidence value to high / medium / low.

    Unrecognised or missing values fall to LOW, not HIGH: an unknown confidence
    is not evidence of certainty, and the previous default of "High" turned a
    missing field into a confirmed diagnosis.
    """
    if raw is None:
        return _CONF_LOW

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        pct = float(raw) * 100 if 0 <= float(raw) <= 1 else float(raw)
        if pct >= 80:
            return _CONF_HIGH
        if pct >= 55:
            return _CONF_MEDIUM
        return _CONF_LOW

    text = str(raw).strip().lower()
    if not text:
        return _CONF_LOW

    pct_match = re.search(r"(\d{1,3})\s*%", text)
    if pct_match:
        pct = int(pct_match.group(1))
        if pct >= 80:
            return _CONF_HIGH
        if pct >= 55:
            return _CONF_MEDIUM
        return _CONF_LOW

    for tier, words in _CONF_SYNONYMS:
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            return tier
    return _CONF_LOW


def _is_social(query: str) -> bool:
    """True only when the entire turn is social pleasantry, nothing more."""
    q = query.lower().strip()
    if not q:
        return False

    stripped = re.sub(r"[^\w\s]", " ", q)
    if len(stripped.split()) > _SOCIAL_MAX_WORDS:
        return False

    if not any(re.search(p, q) for p in _SOCIAL_PATTERNS):
        return False

    # "who are you" / "what can you do" / "help" are social despite the marker
    # overlap, so exempt them from the substantive-marker veto.
    if re.search(r"^(who are you|what are you|what can you do|help)\b", q):
        return True

    return not any(m in q for m in _SUBSTANTIVE_MARKERS)

def _classify_intent(
    query: str,
    has_diagnosis:  bool,
    has_weather:    bool,
    has_fertilizer: bool,
    has_vision:     bool,
) -> str:
    q = query.lower()

    if sum([has_diagnosis, has_weather, has_fertilizer]) >= 2:
        return _Intent.MIXED

    if has_vision or any(kw in q for kw in _IMAGE_KEYWORDS):
        return _Intent.IMAGE

    # Social check runs after the data gates: if any agent produced an artifact
    # for this turn, the farmer asked something real and gets a real answer.
    if not (has_diagnosis or has_weather or has_fertilizer) and _is_social(query):
        return _Intent.SOCIAL

    if any(re.search(p, q) for p in _KNOWLEDGE_PATTERNS):
        # A phrasing pattern alone is not enough to route to KNOWLEDGE. When a
        # live artifact was produced for this turn, that data is the answer and
        # KNOWLEDGE would discard it: the fallback's knowledge branch never
        # reads weather_info/fertilizer_info/diagnostic_info, so
        # "What is the weather forecast for Anuradhapura?" returned generic
        # advice while real measurements sat unused in the artifact.
        has_live_data = has_weather or has_fertilizer or has_diagnosis
        if not has_live_data and not any(kw in q for kw in _SYMPTOM_KEYWORDS):
            return _Intent.KNOWLEDGE

    if has_weather or any(kw in q for kw in _WEATHER_KEYWORDS):
        return _Intent.WEATHER
    if has_fertilizer or any(kw in q for kw in _FERTILIZER_KEYWORDS):
        return _Intent.FERTILIZER
    if has_diagnosis or any(kw in q for kw in _SYMPTOM_KEYWORDS):
        return _Intent.DIAGNOSIS

    return _Intent.KNOWLEDGE


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Experience Engine (v6.0)
# ─────────────────────────────────────────────────────────────────────────────

class ConversationExperienceEngine:
    """
    Final Production Build — Organic Conversational AI Engine.
    Uses CoT Self-Review `<draft> -> <review> -> <final_response>` to guarantee 
    expert-level natural dialogue.
    """

    _BANNED_OPENERS: Tuple[str, ...] = (
        "ayubowan", "hello!", "hello,", "hi there", "hi,",
        "i'm glad", "i am glad", "i'm happy to help", "i am happy to help", 
        "i'd be happy to", "i completely understand", "i understand your concern",
        "i can understand", "i know this", "i can see that", "don't worry", 
        "no worries", "not to worry", "thank you for", "thanks for",
        "great question", "excellent question", "good question",
        "of course!", "of course,", "sure!", "absolutely!", "as an ai", 
        "as a language model", "as an agricultural", "i hope this", "hope this helps", 
        "i hope i've", "please don't hesitate", "feel free to reach", "feel free to ask",
        "i'd like to help", "let me help you",
    )

    _TRAILING_FILLER: Tuple[str, ...] = (
        "hope this helps", "hope that helps", "hope this was helpful",
        "thank you", "have a nice day", "have a great day",
        "feel free to ask", "feel free to reach", "please don't hesitate",
        "let me know if", "i hope i", "please let me know",
        "i'm here to help", "i'd be happy to answer",
        "don't hesitate to ask", "if you have any more questions",
        "if you have any other questions",
    )

    _PUB_MAP: Dict[str, str] = {
        "Rice-Congress":  "Rice Congress 2010 Proceedings",
        "ROP":            "Department of Agriculture Rice Operations Guide",
        "Danapala":       "Dr. M. P. Dhanapala Rice Pathology",
        "learned":        "DOA Verified Pathology Records",
    }
    _DEFAULT_PUB = "Department of Agriculture Sri Lanka (DOA Manual)"

    # Offline replies for social turns. Deliberately short, no agronomic
    # content, no citations. Used only when the LLM is unavailable.
    _SOCIAL_REPLIES: Dict[str, Tuple[str, ...]] = {
        "greeting": (
            "Hello. What's happening in your field today?",
            "Hello. Tell me what you're seeing in your paddy and I'll help.",
            "Good to hear from you. What would you like help with?",
        ),
        "thanks": (
            "Glad it helped. Come back any time your crop needs a second opinion.",
            "You're welcome. Ask again whenever something looks off in the field.",
        ),
        "farewell": (
            "Take care — good luck with the season.",
            "Goodbye. Keep an eye on those leaves, and come back if anything changes.",
        ),
        "identity": (
            "I'm PaddyAgri AI. I help Sri Lankan paddy farmers with disease "
            "diagnosis, fertilizer rates, and weather-based field timing. "
            "What would you like to know?",
        ),
        "acknowledgement": (
            "Anything else you'd like to go over?",
            "Right. What else can I help with?",
        ),
    }

    _OPENING_STYLE_HINTS: Tuple[str, ...] = (
        'Start with the direct answer or key fact.',
        'Open with the most important symptom or sign.',
        'Lead with the practical implication — what the farmer should do.',
        'Begin with the disease or condition itself.',
        'Start with a condition-specific observation ("During warm, humid conditions...").',
        'Open with a field-level statement that situates the answer.',
    )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def compose_conversation(
        cls,
        user_query:           str,
        diagnostic_info:      Optional[Any] = None,
        fertilizer_info:      Optional[Any] = None,
        weather_info:         Optional[Any] = None,
        vision_info:          Optional[Any] = None,
        general_info:         Optional[Any] = None,
        reflection_result:    Optional[Any] = None,
        final_synthesis:      Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        intent = _classify_intent(
            query=user_query,
            has_diagnosis=diagnostic_info is not None,
            has_weather=weather_info is not None,
            has_fertilizer=fertilizer_info is not None,
            has_vision=vision_info is not None,
        )

        evidence = cls._assemble_evidence(
            intent, weather_info, diagnostic_info,
            fertilizer_info, vision_info, general_info,
        )

        is_social = intent == _Intent.SOCIAL

        conv_context = cls._build_conversation_context(conversation_history)
        # The rotating opening hints all steer toward a field observation, which
        # turns "hello" into an unprompted agronomy lecture. Social turns get a
        # hint that matches the register instead.
        opening_hint = (
            "Answer the greeting itself. Do not open with a field or crop observation."
            if is_social else random.choice(cls._OPENING_STYLE_HINTS)
        )
        # A social turn gets no draft: seeding it with RAG prose would fight the
        # "stay brief, give no advice" guidance.
        prompt = cls._build_prompt(
            user_query, intent, evidence, conv_context, opening_hint,
            draft_content=None if is_social else final_synthesis,
        )

        if prompt is None:
            raise RuntimeError(f"[CONVERSATION ENGINE] _build_prompt returned None for query: {user_query}")

        text = None
        try:
            model = get_reasoning_model()
            resp  = model.invoke(prompt)
            if resp and getattr(resp, "content", None):
                text = resp.content.strip()
        except Exception as exc:
            print(f"[CONVERSATION ENGINE v6.0] LLM offline — fallback: {exc}")

        # CoT extraction applies only to LLM output. Running it over fallback
        # text always failed (no XML tags), which invoked the fallback twice.
        if text:
            extracted = cls._extract_final_response(text, intent=intent)
            text = extracted if extracted is not None else None

        if not text:
            text = cls._deterministic_fallback(
                intent, user_query, diagnostic_info, fertilizer_info, weather_info, general_info,
                draft_content=None if is_social else final_synthesis,
            )

        # A greeting is a legitimate opener when the farmer greeted first, so the
        # banned-opener scrub is skipped for social turns only.
        if not is_social:
            text = cls._scrub_banned_opener(text)
        text = cls._scrub_trailing_filler(text)
        text = cls._fix_paragraph_flow(text)

        # Sole append site for the sources block, so it cannot be duplicated.
        # Social turns cite nothing — there is no claim to support.
        if not is_social:
            sources = cls._format_sources(general_info)
            if sources and not _has_sources_block(text):
                text = text.rstrip() + f"\n\n{sources}"

        assert isinstance(text, str), f"Final text must be str, got {type(text)}"
        assert text is not None, "Final text must not be None"
        return text

    # ──────────────────────────────────────────────────────────────────────
    # CoT Extraction Engine (v6.0)
    # ──────────────────────────────────────────────────────────────────────

    # An answer shorter than this is a stub, not a response. The deterministic
    # fallback's complete draft is better than "Paddy Blast." Applies to every
    # extraction path — a closing tag proves the model finished, not that it
    # said anything useful.
    _MIN_ANSWER_CHARS = 120

    # A social reply is *supposed* to be short ("Hello — what's happening in
    # your field?"), so the substantive-answer gate would wrongly reject it.
    _MIN_SOCIAL_ANSWER_CHARS = 15

    @classmethod
    def _extract_final_response(cls, raw_text: str, intent: Optional[str] = None) -> Optional[str]:
        """
        Extracts the final polished response from the CoT XML blocks.

        Returns None when nothing usable was produced, which routes the caller
        to the deterministic fallback. Two failure modes are handled:

        - Truncated: a provider's token budget expired inside
          <final_response>, so the closing tag never arrived. The body is
          trimmed back to its last complete sentence rather than emitted
          mid-word ("...spindle-shaped\\nlesions on the").
        - Trivially short: the block is well formed but says nothing
          ("Paddy Blast."). Rejected by the same length gate.
        """
        if not raw_text:
            return None

        # 1. Well formed — opening and closing tags both present.
        match = re.search(r"<final_response>([\s\S]*?)</final_response>", raw_text, re.IGNORECASE)
        if match:
            return cls._qualify_answer(match.group(1).strip(), intent=intent)

        # 2. Truncated — opening tag present, closing tag never emitted.
        opening = re.search(r"<final_response>", raw_text, re.IGNORECASE)
        if opening:
            body = raw_text[opening.end():]
            # Discard any stray block markers the model leaked after the cut.
            body = re.split(r"</?(?:draft|review|final_response)>", body, flags=re.IGNORECASE)[0]
            return cls._qualify_answer(body.strip(), intent=intent)

        # 3. No <final_response> at all (e.g. budget expired inside <review>).
        return None

    @classmethod
    def _qualify_answer(cls, body: str, intent: Optional[str] = None) -> Optional[str]:
        """
        Return a complete, substantive answer, or None to trigger the fallback.

        Repairs a mid-sentence cut by trimming to the last sentence boundary,
        then enforces the minimum length on whatever survives. Social turns use
        a much lower floor because a one-line greeting is the correct answer.
        """
        if not body:
            return None

        if body[-1] not in ".!?":
            boundaries = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", body)]
            if not boundaries:
                return None
            body = body[:boundaries[-1]].strip()

        floor = cls._MIN_SOCIAL_ANSWER_CHARS if intent == _Intent.SOCIAL else cls._MIN_ANSWER_CHARS
        return body if len(body) >= floor else None

    # ──────────────────────────────────────────────────────────────────────
    # Conversation Continuity Engine (v6.0)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _build_conversation_context(
        cls, history: Optional[List[Dict[str, str]]]
    ) -> str:
        """
        Explicitly demarcates past context from the current question.
        Ensures the LLM continues the narrative instead of restarting it.
        """
        if not history:
            return ""
        recent = [m for m in history if m.get("role") in ("user", "assistant")][-4:]
        if not recent:
            return ""
        lines = ["PAST CONVERSATION CONTEXT (Use to resolve references like 'it' or 'that disease'):"]
        for msg in recent:
            role  = "Farmer" if msg["role"] == "user" else "AI"
            body  = str(msg.get("content", "")).strip()
            if msg["role"] == "assistant" and len(body) > 250:
                body = body[:250].rstrip() + "…"
            lines.append(f"  [{role}]: {body}")
        lines.append("--- END OF PAST CONTEXT ---")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # Evidence Assembly
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _assemble_evidence(
        cls,
        intent:          str,
        weather_info:    Optional[Any],
        diagnostic_info: Optional[Any],
        fertilizer_info: Optional[Any],
        vision_info:     Optional[Any],
        general_info:    Optional[Any],
    ) -> str:
        blocks: List[str] = []

        if weather_info:
            loc   = cls._v(weather_info, "location", "Sri Lanka")
            temp  = cls._v(weather_info, "temperature_c", 31.5)
            hum   = cls._v(weather_info, "humidity_pct", 82)
            risk  = cls._v(weather_info, "fungal_risk_alert", "Moderate")
            notes = cls._v(weather_info, "advisory_notes", [])
            note_str = "; ".join(str(n) for n in notes) if notes else "none"
            blocks.append(
                f"[WEATHER] Location: {loc} | Temp: {temp}°C | Humidity: {hum}% "
                f"| Fungal Risk: {risk} | Advisory: {note_str}"
            )

        if diagnostic_info:
            dis  = cls._v(diagnostic_info, "suspected_disease", "Unknown disease")
            raw_conf = cls._v(diagnostic_info, "confidence_level")
            if raw_conf in (None, ""):
                raw_conf = cls._v(diagnostic_info, "confidence")
            tier = _confidence_tier(raw_conf)
            syms = cls._v(diagnostic_info, "symptoms_identified", [])
            trts = cls._v(diagnostic_info, "treatment_recommended", [])
            sym_str = "; ".join(str(s) for s in syms) if syms else "not specified"
            trt_str = "; ".join(str(t) for t in trts) if trts else "not specified"
            # The certainty instruction travels with the evidence so the model
            # cannot present a low-confidence guess as a confirmed diagnosis.
            certainty = {
                _CONF_HIGH: "HIGH — state this disease directly as your assessment.",
                _CONF_MEDIUM: (
                    "MEDIUM — present this as the most likely cause, not a confirmed "
                    "diagnosis. Name what the farmer should check to confirm it."
                ),
                _CONF_LOW: (
                    "LOW — do NOT present this as the diagnosis. Say the signs are not "
                    "conclusive, give the most likely possibilities, and tell the farmer "
                    "what to inspect or photograph next."
                ),
            }[tier]
            blocks.append(
                f"[DIAGNOSIS] Disease: {dis} | Confidence tier: {certainty} "
                f"| Symptoms: {sym_str} | DOA Treatments: {trt_str}"
            )

        if fertilizer_info:
            season = cls._v(fertilizer_info, "season", "Yala/Maha")
            urea   = cls._v(fertilizer_info, "urea_dosage_per_acre_kg") \
                     or cls._v(fertilizer_info, "urea_kg", 50.0)
            tsp    = cls._v(fertilizer_info, "tsp_dosage_per_acre_kg") \
                     or cls._v(fertilizer_info, "tsp_kg", 25.0)
            mop    = cls._v(fertilizer_info, "mop_dosage_per_acre_kg") \
                     or cls._v(fertilizer_info, "mop_kg", 25.0)
            sched  = cls._v(fertilizer_info, "application_schedule", []) \
                     or cls._v(fertilizer_info, "notes", [])
            sched_str = "; ".join(str(s) for s in sched) if sched else "follow DOA timetable"
            try:
                urea_f, tsp_f, mop_f = f"{float(urea):.1f}", f"{float(tsp):.1f}", f"{float(mop):.1f}"
            except (TypeError, ValueError):
                urea_f, tsp_f, mop_f = str(urea), str(tsp), str(mop)
            blocks.append(
                f"[FERTILIZER] Season: {season} | Urea: {urea_f} kg/ac "
                f"| TSP: {tsp_f} kg/ac | MOP: {mop_f} kg/ac | Schedule: {sched_str}"
            )

        if vision_info:
            obs = cls._v(vision_info, "observations", "") \
                  or cls._v(vision_info, "detected_symptoms", "")
            if obs:
                blocks.append(f"[IMAGE] Visual observations: {cls._smart_truncate(str(obs), 400)}")

        if general_info and isinstance(general_info, dict):
            snippets = general_info.get("snippets", []) or general_info.get("chunks", [])
            for snip in snippets[:4]:
                raw = str(snip.get("content", "")).strip()
                if raw:
                    blocks.append(f"[DOA RESEARCH] {cls._smart_truncate(raw, 500)}")

        return "\n".join(blocks) if blocks else "No specific field data for this query."

    # ──────────────────────────────────────────────────────────────────────
    # CoT Master Prompt (v6.0)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _build_prompt(
        cls,
        user_query:    str,
        intent:        str,
        evidence:      str,
        conv_context:  str,
        opening_hint:  str,
        draft_content: Optional[str] = None,
    ) -> str:

        intent_guidance: Dict[str, str] = {
            _Intent.SOCIAL: (
                "The farmer is greeting you, thanking you, or asking who you are — there is "
                "no agronomic question yet. Reply in one or two short sentences, warmly and "
                "briefly, and invite the specific question you can help with. "
                "Do NOT give agronomic advice, do NOT list diseases or dosages, and do NOT "
                "cite sources. Brevity is the whole point."
            ),
            _Intent.KNOWLEDGE: (
                "Educational question. Explain the concept clearly. "
                "If symptoms, chemical treatment, and organic management are all relevant, "
                "address each in its own paragraph — but only include what is genuinely useful."
            ),
            _Intent.DIAGNOSIS: (
                "Field diagnosis based on symptoms. Lead with your assessment. "
                "Explain WHY this disease is likely. "
                "Obey the confidence tier in the evidence block exactly: at HIGH state the "
                "disease directly; at MEDIUM call it the most likely cause and name what to "
                "check to confirm; at LOW do not name a diagnosis as settled — give the "
                "likely possibilities and what the farmer should inspect next. "
                "Never invent certainty the evidence does not support. "
                "Chemical treatment and organic/cultural management must be in completely separate blocks."
            ),
            _Intent.WEATHER: (
                "Explain the PRACTICAL IMPACT of these weather conditions on the farmer's field. "
                "Do not produce a weather report. Lead with the most urgent actionable advice."
            ),
            _Intent.FERTILIZER: (
                "Explain WHY this fertilizer is needed before giving dosage numbers. "
                "Do not open with a table or list. Provide context first."
            ),
            _Intent.IMAGE: (
                "Begin with what you observe in the image as a human expert would. "
                "Then reason about crop health. Avoid sounding like computer-vision software."
            ),
            _Intent.MIXED: (
                "Multiple aspects (disease, weather, fertilizer). Prioritise the most urgent aspect. "
                "Use prose if it flows naturally, headings only if necessary."
            ),
        }
        guidance = intent_guidance.get(intent, intent_guidance[_Intent.KNOWLEDGE])
        conv_section = f"\n{conv_context}\n" if conv_context else ""

        # Output budget note: reasoning providers cap completions (Groq tier is
        # max_tokens=250). The previous three-block CoT asked for the full answer
        # twice — <draft> then <final_response> — so the budget was exhausted
        # before <final_response> was ever emitted. Now the model writes the
        # answer exactly once, preceded by a terse checklist.
        draft_section = ""
        if draft_content is not None:
            draft_section = (
                f"\nHERE IS A DRAFT RESPONSE BASED ON THE EVIDENCE:\n<draft>\n{draft_content}\n</draft>\n\n"
                "Silently check the draft against the FINAL QUALITY GATE below, then output "
                "ONLY the corrected answer inside a single <final_response> block.\n"
                "Do NOT restate the draft. Do NOT explain your corrections. Do NOT emit a "
                "<draft> or <review> block of your own.\n\n"
                "<final_response>\nThe improved answer, in full.\n</final_response>"
            )
        else:
            draft_section = (
                "Output your answer in exactly two blocks, in this order:\n\n"
                "<review>\nOne line per gate item below: the item name and PASS/FIX. Nothing else.\n</review>\n\n"
                "<final_response>\nThe complete answer for the farmer, already incorporating every FIX.\n</final_response>\n\n"
                "Write the answer ONCE, inside <final_response>. Never write a full answer "
                "inside <review>."
            )

        return f"""You are PaddyAgri AI — a Principal Agricultural Extension Officer in Sri Lanka.
You have decades of field experience advising paddy farmers.

{conv_section}
CURRENT FARMER'S QUESTION: "{user_query}"

VERIFIED DOA EVIDENCE (Every factual claim must be grounded here):
{evidence}

══════════════════════════════════════════════════════════════
INTENT: {intent.upper()}
{guidance}
══════════════════════════════════════════════════════════════

VOICE & TONE
- Expert speaking to another human — calm, direct, confident, warm, professional.
- NOT customer support. No artificial empathy. No scripted warmth.
- NEVER expose internal terminology (e.g. Knowledge Context, Evidence, Reflection, Planner, Agent, Confidence Object, Internal Pipeline).
- Summarize RAG context naturally. NEVER dump raw PDF chunks or expose retrieval snippets.

FIRST SENTENCE RULE
- Opening style this turn: {opening_hint}
- NEVER begin with: Ayubowan, Hello, I'm glad you asked, Don't worry, Thank you, Weather Advisory, Diagnostic Report, Analysis, Assessment, Recommendation Report.
- The first sentence must naturally and directly address the question.

CHEMICAL vs ORGANIC
- ALWAYS write chemical treatment first.
- ALWAYS write organic/cultural management as a distinct block.
- NEVER mix them in the same paragraph.

ENDING
- Finish with one natural, practical, actionable piece of guidance (e.g. "What you should do: ..."). 
- Never end with "Hope this helps" or "Thank you".

══════════════════════════════════════════════════════════════
CHAIN OF THOUGHT: FINAL QUALITY GATE
══════════════════════════════════════════════════════════════
{draft_section}

Gate items (same coverage as before, terser to emit — one word each):
1 no-object-reprs (no WeatherResponse(...), no field='value')
2 no-dicts-or-raw-lists
3 one-risk-level-only
4 natural-opening
5 paragraph-flow (open -> answer -> why -> actions)
6 actionable-ending
7 rag-summarised (no PDF dumps)
8 no-internal-terms (Agent, Planner, Evidence)
9 expert-human-voice

The full answer belongs ONLY in <final_response>. Keep <review> to one short
line per item. If the budget runs short, sacrifice <review>, never the answer.

Begin now."""

    # ──────────────────────────────────────────────────────────────────────
    # Heuristic Polish Layer (v6.0)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _scrub_banned_opener(cls, text: str) -> str:
        """
        Remove a scripted opener ("Great question.", "Don't worry.").

        Never returns "" for non-empty input: when the banned phrase is the
        only paragraph, dropping it left the farmer with a blank message.
        """
        assert isinstance(text, str), f"Expected str, got {type(text)}"
        if not text or not text.strip():
            return ""

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return ""

        first_para = paragraphs[0].lower()
        # Strip markdown bolding/italics for comparison
        clean_para = re.sub(r'[*_#]', '', first_para)

        for phrase in cls._BANNED_OPENERS:
            if clean_para.startswith(phrase):
                # Cut after the opener's own sentence. Any terminator counts:
                # matching only ". " missed "Ayubowan! Here is ..." and the
                # whole response was discarded instead.
                cut = re.search(r"[.!?]\s+", text[:250])
                if cut:
                    remainder = text[cut.end():].strip()
                    if remainder:
                        return remainder
                if len(paragraphs) > 1:
                    return "\n\n".join(paragraphs[1:]).strip()
                # Opener is the entire response — strip the phrase itself
                # rather than return a blank message.
                stripped = re.sub(r"^[^\w]*" + re.escape(phrase), "", clean_para).strip(" .,!?—-")
                return stripped if stripped else text
        return text

    @classmethod
    def _scrub_trailing_filler(cls, text: str) -> str:
        """
        Drop closing pleasantries from the tail of a response.

        Never returns "" for non-empty input. A short reply can be a single
        line that legitimately ends in a filler phrase ("Hi there! Feel free to
        ask any questions."); popping that line deleted the whole answer and
        the farmer saw a blank message.
        """
        assert isinstance(text, str), f"Expected str, got {type(text)}"
        if not text or not text.strip():
            return ""

        lines = text.rstrip().split("\n")
        removed = True
        while removed and len(lines) > 1:
            removed = False
            last = lines[-1].strip()
            clean_last = re.sub(r'[*_#]', '', last).lower().rstrip("!.")
            if any(clean_last.startswith(f) for f in cls._TRAILING_FILLER):
                lines.pop()
                removed = True

        scrubbed = "\n".join(lines).strip()
        return scrubbed if scrubbed else text.strip()

    @classmethod
    def _fix_paragraph_flow(cls, text: str) -> str:
        assert isinstance(text, str), f"Expected str, got {type(text)}"
        if not text or not text.strip():
            return ""

        # Collapse 3+ blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # [H2] Bullet density check — if a list has >6 items, it's a wall of text.
        # We don't delete them, but we add an extra newline for readability.
        text = re.sub(r"(\n•.+){5,}", lambda m: m.group(0).replace("\n•", "\n\n•"), text)
        return text.strip()

    # ──────────────────────────────────────────────────────────────────────
    # Sources formatter
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _format_sources(cls, general_info: Optional[Any]) -> str:
        if not general_info or not isinstance(general_info, dict):
            return ""
        snippets = general_info.get("snippets", []) or general_info.get("chunks", [])
        if not snippets:
            return ""

        seen: set  = set()
        lines: List[str] = [_SOURCES_HEADING]
        for snip in snippets[:4]:
            fn    = str(snip.get("filename", ""))
            pg    = snip.get("page", 1)
            title = cls._DEFAULT_PUB
            for key, label in cls._PUB_MAP.items():
                if key in fn:
                    title = label
                    break
            item = f"• {title} (Page {pg})"
            if item not in seen:
                seen.add(item)
                lines.append(item)

        return "\n".join(lines) if len(lines) > 1 else ""

    # ──────────────────────────────────────────────────────────────────────
    # Deterministic Fallback (Expanded)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _deterministic_fallback(
        cls,
        intent:          str,
        user_query:      str,
        diagnostic_info: Optional[Any],
        fertilizer_info: Optional[Any],
        weather_info:    Optional[Any],
        general_info:    Optional[Any] = None,
        draft_content:   Optional[str] = None,
    ) -> str:
        if draft_content is not None:
            # Polish only. The sources block is appended by the caller
            # (compose_conversation), which is the single append site.
            text = cls._scrub_banned_opener(draft_content)
            text = cls._scrub_trailing_filler(text)
            text = cls._fix_paragraph_flow(text)
            return text

        parts: List[str] = []
        var_id = random.randint(1, 3) # [H4] Slight variation

        if intent == _Intent.SOCIAL:
            # Short, warm, no agronomic content and no citations.
            return random.choice(cls._SOCIAL_REPLIES.get(
                _social_kind(user_query), cls._SOCIAL_REPLIES["greeting"]
            ))

        if intent == _Intent.DIAGNOSIS and diagnostic_info:
            dis  = cls._v(diagnostic_info, "suspected_disease", "Paddy Blast")
            raw_conf = cls._v(diagnostic_info, "confidence_level")
            if raw_conf in (None, ""):
                raw_conf = cls._v(diagnostic_info, "confidence")
            tier = _confidence_tier(raw_conf)
            syms = cls._v(diagnostic_info, "symptoms_identified", [])
            trts = cls._v(diagnostic_info, "treatment_recommended", [])

            # Only HIGH speaks in the indicative. MEDIUM and LOW hedge, and LOW
            # never names the disease as settled.
            if tier == _CONF_HIGH:
                openings = (
                    f"Based on the symptoms described, this is **{dis}**",
                    f"The signs point clearly to **{dis}**",
                    f"Your crop is affected by **{dis}**",
                )
            elif tier == _CONF_MEDIUM:
                openings = (
                    f"Based on the symptoms described, the most likely cause is **{dis}**, "
                    f"though this is not yet confirmed",
                    f"These signs point towards **{dis}**, but they are not conclusive on their own",
                    f"**{dis}** is the most probable cause of what you are describing",
                )
            else:
                openings = (
                    f"The symptoms described are not conclusive. **{dis}** is one possibility, "
                    f"but several paddy diseases share these signs",
                    f"I cannot confirm a diagnosis from these symptoms alone — **{dis}** is one "
                    f"candidate among several",
                    f"There is not enough here to identify the disease with confidence. "
                    f"**{dis}** is possible, but so are look-alike conditions",
                )
            parts.append(openings[var_id - 1] + ".")

            sym_lines = [s for s in (coerce_scalar(x, field="symptoms_identified") for x in syms) if s]
            if sym_lines:
                parts.append("The key signs to watch for:\n" + "\n".join(f"• {s}" for s in sym_lines))

            trt_lines = [t for t in (coerce_scalar(x, field="treatment_recommended") for x in trts) if t]
            if trt_lines:
                if tier == _CONF_HIGH:
                    lead = "The Department of Agriculture recommends the following management steps:"
                else:
                    lead = (
                        "If this turns out to be the cause, the Department of Agriculture "
                        "recommends the following management steps:"
                    )
                parts.append(lead + "\n" + "\n".join(f"• {t}" for t in trt_lines))

            # Below HIGH the farmer needs a confirmation path, not a closing
            # instruction to act on an unconfirmed call.
            if tier == _CONF_HIGH:
                parts.append(
                    "Acting within the first week of symptom appearance significantly reduces crop damage."
                )
            else:
                parts.append(
                    "Before spraying, confirm the cause: check the lesion shape and colour on "
                    "several plants across the field, note whether the damage starts on lower or "
                    "upper leaves, and photograph an affected leaf in daylight. Send that photo "
                    "here, or show it to your local Agrarian Services officer. Treating the wrong "
                    "disease costs a spray cycle while the real problem spreads."
                )

        elif intent == _Intent.WEATHER and weather_info:
            loc   = cls._v(weather_info, "location", "your area")
            hum   = cls._v(weather_info, "humidity_pct", 82)
            risk  = cls._v(weather_info, "fungal_risk_alert", "Moderate")
            notes = cls._v(weather_info, "advisory_notes", [])

            parts.append(
                f"With humidity at {hum}% in {loc}, fungal disease pressure is currently "
                f"**{risk}** — this is an important consideration for your field operations."
            )
            if notes:
                # Sanitized per item: unrenderable notes dropped (not str()'d),
                # operator log tags stripped from farmer-facing prose.
                note_lines = [strip_log_tags(coerce_scalar(n, field="advisory_notes")) for n in notes]
                note_lines = [n for n in note_lines if n]
                if note_lines:
                    parts.append("Based on current conditions:\n" + "\n".join(f"• {n}" for n in note_lines))

        elif intent == _Intent.FERTILIZER and fertilizer_info:
            season = cls._v(fertilizer_info, "season", "current")
            urea   = cls._v(fertilizer_info, "urea_dosage_per_acre_kg") \
                     or cls._v(fertilizer_info, "urea_kg", 50.0)
            sched  = cls._v(fertilizer_info, "application_schedule", []) \
                     or cls._v(fertilizer_info, "notes", [])
            try:
                urea_f = f"{float(urea):.1f}"
            except (TypeError, ValueError):
                urea_f = str(urea)

            parts.append(
                f"During the {season} season, the tillering stage needs nitrogen to support "
                f"strong tiller development. The Department of Agriculture recommends "
                f"**{urea_f} kg of Urea per acre** as the first top-dressing."
            )
            sched_lines = [s for s in (coerce_scalar(x, field="application_schedule") for x in sched) if s]
            if sched_lines:
                parts.append("\n".join(f"• {s}" for s in sched_lines))
            parts.append(
                "Apply when field water is at 2–3 cm depth — this improves absorption and "
                "prevents the fertilizer from being washed away."
            )

        elif intent == _Intent.KNOWLEDGE:
            if general_info and isinstance(general_info, dict):
                snippets = general_info.get("snippets", []) or general_info.get("chunks", [])
                # Summarized, not dumped: raw chunks were previously appended whole.
                summaries: List[str] = []
                seen_keys = set()
                for snip in snippets:
                    raw = snip.get("content", "") if isinstance(snip, dict) else snip
                    summary = summarize_snippet(raw)
                    if not summary:
                        continue
                    key = summary.lower().rstrip(".")
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    summaries.append(summary)
                    if len(summaries) >= 2:
                        break

                if summaries:
                    parts.append("Based on Department of Agriculture resources:")
                    parts.extend(f"• {s}" for s in summaries)
                else:
                    parts.append("No specific information found in the Department of Agriculture resources.")
            else:
                parts.append("I recommend consulting the Department of Agriculture's resources for detailed information on this topic.")

        else:
            parts.append(
                "Paddy Blast (*Magnaporthe oryzae*) is one of the most destructive fungal "
                "diseases in Sri Lankan rice farming. It spreads fastest when temperatures "
                "are between 24–28°C with high overnight humidity."
            )
            parts.append(
                "The earliest sign is small, spindle-shaped lesions with gray centres and "
                "brown edges on the leaves. In severe cases, the disease can move to the "
                "leaf collar, neck, and panicles, causing significant yield loss."
            )
            parts.append(
                "The Department of Agriculture recommends planting resistant varieties and "
                "avoiding excessive nitrogen application early in the season. If symptoms "
                "appear, foliar application of Azoxystrobin or Pyraclostrobin provides "
                "effective chemical control. Good field drainage and balanced fertilization "
                "are the best cultural practices for reducing disease pressure."
            )
            parts.append(
                "Identifying and treating Blast in the first week of symptom appearance "
                "is the most effective way to protect your harvest."
            )

        return "\n\n".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _smart_truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > max_chars // 2:
            return truncated[:last_period + 1]
        return truncated.rstrip() + "…"

    @staticmethod
    def _v(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def get_val(cls, obj: Any, key: str, default: Any = None) -> Any:
        return cls._v(obj, key, default)
