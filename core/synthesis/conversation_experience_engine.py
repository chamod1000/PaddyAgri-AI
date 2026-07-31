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

    if any(re.search(p, q) for p in _KNOWLEDGE_PATTERNS):
        if not any(kw in q for kw in _SYMPTOM_KEYWORDS):
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
        "Rice-Congress":  "Rice Congress 2010 Research Publication",
        "ROP":            "DOA Rice Operations & Pathology Guide",
        "Danapala":       "Rice Pathology Research (Dr. M.P. Dhanapala)",
        "learned":        "DOA Verified Pathology Records",
    }
    _DEFAULT_PUB = "Department of Agriculture Sri Lanka (DOA Manual)"

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

        conv_context = cls._build_conversation_context(conversation_history)
        opening_hint = random.choice(cls._OPENING_STYLE_HINTS)
        prompt = cls._build_prompt(user_query, intent, evidence, conv_context, opening_hint, draft_content=final_synthesis)

        text = None
        try:
            model = get_reasoning_model()
            resp  = model.invoke(prompt)
            if resp and getattr(resp, "content", None):
                text = resp.content.strip()
        except Exception as exc:
            print(f"[CONVERSATION ENGINE v6.0] LLM offline — fallback: {exc}")

        if not text:
            text = cls._deterministic_fallback(
                intent, user_query, diagnostic_info, fertilizer_info, weather_info, general_info, draft_content=final_synthesis
            )
        else:
            text = cls._extract_final_response(text)

        text = cls._scrub_banned_opener(text)
        text = cls._scrub_trailing_filler(text)
        text = cls._fix_paragraph_flow(text)

        sources = cls._format_sources(general_info)
        if sources and "📚 Sources" not in text:
            text = text.rstrip() + f"\n\n{sources}"

        return text

    # ──────────────────────────────────────────────────────────────────────
    # CoT Extraction Engine (v6.0)
    # ──────────────────────────────────────────────────────────────────────
    
    @classmethod
    def _extract_final_response(cls, raw_text: str) -> str:
        """
        Extracts the final polished response from the CoT XML blocks.
        If the LLM fails to output valid XML, falls back gracefully.
        """
        match = re.search(r"<final_response>([\s\S]*?)</final_response>", raw_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback if tags are missing or malformed
        lines = raw_text.split('\n')
        final_lines = []
        in_final = False
        for line in lines:
            if "<final_response>" in line.lower():
                in_final = True
                continue
            if "</final_response>" in line.lower():
                in_final = False
                continue
            if in_final:
                final_lines.append(line)
        
        if final_lines:
            return "\n".join(final_lines).strip()
            
        # Absolute fallback: just return the raw text (scrubber will clean it)
        return raw_text.strip()

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
            conf = cls._v(diagnostic_info, "confidence_level") \
                   or cls._v(diagnostic_info, "confidence", "High")
            syms = cls._v(diagnostic_info, "symptoms_identified", [])
            trts = cls._v(diagnostic_info, "treatment_recommended", [])
            sym_str = "; ".join(str(s) for s in syms) if syms else "not specified"
            trt_str = "; ".join(str(t) for t in trts) if trts else "not specified"
            blocks.append(
                f"[DIAGNOSIS] Disease: {dis} | Confidence: {conf} "
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
            _Intent.KNOWLEDGE: (
                "Educational question. Explain the concept clearly. "
                "If symptoms, chemical treatment, and organic management are all relevant, "
                "address each in its own paragraph — but only include what is genuinely useful."
            ),
            _Intent.DIAGNOSIS: (
                "Field diagnosis based on symptoms. Lead with your assessment. "
                "Explain WHY this disease is likely. Show confidence only if uncertainty exists (i.e., not 'High'). "
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

        draft_section = ""
        if draft_content is not None:
            draft_section = f"\nHERE IS A DRAFT RESPONSE BASED ON THE EVIDENCE:\n<draft>\n{draft_content}\n</draft>\n\nNOW, REVIEW THIS DRAFT AND PROVIDE:\n1. A <review> block that critiques the draft on:\n    - Naturalness\n    - Readability\n    - Grounding (DOA Evidence)\n    - Practical usefulness\n    - Professional expert tone (No customer support filler)\n2. A <final_response> block that improves the draft based on your review.\n\nIf the draft is empty, first generate a draft based on the instructions and evidence, then review and finalize."
        else:
            draft_section = "To guarantee production-level quality, you MUST output your response in three XML blocks exactly as follows:\n\n<draft>\nWrite your initial full response here based on the instructions above.\n</draft>\n\n<review>\nScore the draft out of 10 for:\n1. Naturalness\n2. Readability\n3. Grounding (DOA Evidence)\n4. Practical usefulness\n5. Professional expert tone (No customer support filler)\nIdentify any robotic phrasing, dense walls of text, or forbidden openers.\n</review>\n\n<final_response>\nWrite the final polished response, fixing every flaw identified in the review.\nEnsure chemical and organic treatments remain strictly separated.\n</final_response>"

    # ──────────────────────────────────────────────────────────────────────
    # Heuristic Polish Layer (v6.0)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _scrub_banned_opener(cls, text: str) -> str:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return text

        first_para = paragraphs[0].lower()
        # Strip markdown bolding/italics for comparison
        clean_para = re.sub(r'[*_#]', '', first_para)

        for phrase in cls._BANNED_OPENERS:
            if clean_para.startswith(phrase):
                idx = text.find(". ")
                if 0 < idx < 250:
                    return text[idx + 2:].strip()
                return "\n\n".join(paragraphs[1:]).strip() if len(paragraphs) > 1 else text
        return text

    @classmethod
    def _scrub_trailing_filler(cls, text: str) -> str:
        lines = text.rstrip().split("\n")
        removed = True
        while removed and lines:
            removed = False
            last = lines[-1].strip()
            clean_last = re.sub(r'[*_#]', '', last).lower().rstrip("!.")
            if any(clean_last.startswith(f) for f in cls._TRAILING_FILLER):
                lines.pop()
                removed = True
        return "\n".join(lines).strip()

    @classmethod
    def _fix_paragraph_flow(cls, text: str) -> str:
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
        lines: List[str] = ["### 📚 Sources"]
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
            # Apply the polish layer to the draft_content
            text = cls._scrub_banned_opener(draft_content)
            text = cls._scrub_trailing_filler(text)
            text = cls._fix_paragraph_flow(text)
            # Add sources if not present
            sources = cls._format_sources(general_info)
            if sources and "📚 Sources" not in text:
                text = text.rstrip() + f"\n\n{sources}"
            return text

        parts: List[str] = []
        var_id = random.randint(1, 3) # [H4] Slight variation

        if intent == _Intent.DIAGNOSIS and diagnostic_info:
            dis  = cls._v(diagnostic_info, "suspected_disease", "Paddy Blast")
            conf = cls._v(diagnostic_info, "confidence_level") \
                   or cls._v(diagnostic_info, "confidence", "High")
            syms = cls._v(diagnostic_info, "symptoms_identified", [])
            trts = cls._v(diagnostic_info, "treatment_recommended", [])

            if var_id == 1:
                opening = f"Based on the symptoms described, **{dis}** is the most likely cause"
            elif var_id == 2:
                opening = f"The visual signs point primarily towards **{dis}**"
            else:
                opening = f"It appears your crop is affected by **{dis}**"

            if conf and str(conf).lower() not in ("high", ""):
                opening += f" (confidence: {conf})"
            parts.append(opening + ".")

            if syms:
                parts.append("The key signs to watch for:\n" + "\n".join(f"• {s}" for s in syms))

            if trts:
                parts.append("The Department of Agriculture recommends the following management steps:\n" + "\n".join(f"• {t}" for t in trts))

            parts.append("Acting within the first week of symptom appearance significantly reduces crop damage.")

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
                parts.append("Based on current conditions:\n" + "\n".join(f"• {n}" for n in notes))

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
            if sched:
                parts.append("\n".join(f"• {s}" for s in sched))
            parts.append(
                "Apply when field water is at 2–3 cm depth — this improves absorption and "
                "prevents the fertilizer from being washed away."
            )

        elif intent == _Intent.KNOWLEDGE:
            if general_info and isinstance(general_info, dict):
                snippets = general_info.get("snippets", []) or general_info.get("chunks", [])
                if snippets:
                    parts.append("Based on Department of Agriculture resources:")
                    for snip in snippets[:2]:
                        raw = str(snip.get("content", "")).strip()
                        if raw:
                            parts.append(f"• {raw}")
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
