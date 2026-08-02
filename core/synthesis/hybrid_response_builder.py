"""
HybridResponseBuilder Module - Version 3.0 Architecture
Renders structured evidence directly via Streamlit UI components (0 LLM tokens)
or invokes LLM synthesis when complex reasoning is required.

Presentation-layer contract (see core/synthesis/text_sanitizer.py):
  - Every value written into the response passes through coerce_scalar().
    An object that cannot be rendered is omitted and logged, never str()'d.
  - Every retrieved DOA chunk passes through summarize_snippet().
    Raw PDF text is never emitted.
"""

from typing import Dict, Any, Tuple, List
from core.evidence.evidence_graph import EvidenceGraph
from core.synthesis.ui_contracts import WeatherCardContract, FertilizerTableContract, MarketGridContract
from core.synthesis.text_sanitizer import coerce_scalar, summarize_snippet, strip_log_tags


class HybridResponseBuilder:
    """Hybrid rendering engine maximizing token efficiency and evidence preservation."""

    @staticmethod
    def _val(v: Any, field: str = "") -> str:
        """
        Render a value as farmer-safe text, or "" if it cannot be rendered safely.

        Delegates to the shared sanitizer. There is deliberately no str(obj)
        fallback: that fallback is what leaked pydantic model reprs such as
        "fungal_risk_alert='HIGH RISK' advisory_notes=[...]" into responses.
        """
        return coerce_scalar(v, field=field)

    @classmethod
    def render_response(cls, evidence_graph: EvidenceGraph) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (rendered_markdown_text, ui_components_dict).
        Merges ALL evidence artifacts (Weather, NPK, Pathology, Market, RAG Findings)
        into a natural summary, preventing raw object leakage.
        """
        ui_components = {}
        rendered_parts = []

        # 1. Weather Intelligence
        weather_art = evidence_graph.get_artifact("weather_intelligence")
        if weather_art:
            loc = cls._val(weather_art.get("location", "Anuradhapura"), "location")
            temp = cls._val(weather_art.get("temperature_c", 28.5), "temperature_c")
            hum = cls._val(weather_art.get("humidity_pct", 82), "humidity_pct")
            alert = cls._val(weather_art.get("fungal_risk_alert", "Moderate"), "fungal_risk_alert")

            w_text = f"Based on the current weather conditions in {loc} "
            if temp and hum:
                w_text += f"(Temperature: {temp}°C, Humidity: {hum}%) "
            if alert:
                w_text += f"the fungal disease risk is considered {alert}."
            else:
                w_text = w_text.rstrip() + "."

            # Advisories are distinct fields; each is rendered once so that no
            # second, contradicting risk level can appear in the same answer.
            advisories = [
                cls._val(weather_art.get("timing_advice", ""), "timing_advice"),
                cls._val(weather_art.get("irrigation_advice", ""), "irrigation_advice"),
            ]
            advisories = [a for a in advisories if a]
            if advisories:
                w_text += " The current recommendation is: " + "; ".join(advisories) + "."

            rendered_parts.append(w_text.strip())

            # Advisory notes carry operator log tags ("[WARNING] ", "[RAIN ALERT] ")
            # and are full sentences. Stripped and bulleted, never comma-joined.
            raw_notes = weather_art.get("advisory_notes", []) or []
            note_lines = []
            for n in raw_notes:
                clean = strip_log_tags(cls._val(n, "advisory_notes"))
                if clean and clean not in note_lines:
                    note_lines.append(clean)
            if note_lines:
                rendered_parts.append(
                    "What this means for your field:\n" + "\n".join(f"• {n}" for n in note_lines)
                )

        # 2. Pathology Diagnosis
        diag_art = evidence_graph.get_artifact("pathology_diagnosis")
        if diag_art:
            disease = cls._val(diag_art.get("suspected_disease", "an unknown condition"), "suspected_disease")
            conf = cls._val(diag_art.get("confidence_level", "High"), "confidence_level")
            treatments = cls._val(diag_art.get("treatment_recommended", []), "treatment_recommended")

            d_text = f"My analysis indicates that the crop is likely affected by {disease}"
            if conf:
                d_text += f" (Confidence: {conf})"
            d_text += "."
            if treatments:
                d_text += f" Recommended management includes: {treatments}."
            rendered_parts.append(d_text)

        # 3. Fertilizer Advisory
        fert_art = evidence_graph.get_artifact("npk_formulation")
        if fert_art:
            season = cls._val(fert_art.get("season", "the current season"), "season")
            urea = cls._val(fert_art.get("urea_kg", 50.0), "urea_kg")
            tsp = cls._val(fert_art.get("tsp_kg", 25.0), "tsp_kg")
            mop = cls._val(fert_art.get("mop_kg", 25.0), "mop_kg")
            notes = cls._val(fert_art.get("notes", []), "notes")

            f_text = (
                f"For fertilizer application during {season}, the Department of Agriculture recommends "
                f"Urea at {urea} kg/acre, TSP at {tsp} kg/acre, and MOP at {mop} kg/acre."
            )
            if notes:
                f_text += f" Please note: {notes}"
            rendered_parts.append(f_text)

        # 4. Market Pricing
        market_art = evidence_graph.get_artifact("market_pricing")
        if market_art:
            dist = cls._val(market_art.get("district", "Islandwide"), "district")
            samba = cls._val(market_art.get("samba_price_lkr", 120.00), "samba_price_lkr")
            nadu = cls._val(market_art.get("nadu_price_lkr", 105.00), "nadu_price_lkr")
            rendered_parts.append(
                f"For your reference, the current Paddy Marketing Board buying prices in {dist} are "
                f"LKR {samba}/kg for Samba and LKR {nadu}/kg for Nadu."
            )

        # 5. Knowledge Base RAG Findings (summarized, never dumped verbatim)
        know_art = evidence_graph.get_artifact("knowledge_retrieval")
        if know_art:
            snippets = know_art.get("snippets", []) or know_art.get("chunks", [])
            summaries = cls._summarize_snippets(snippets, limit=2)
            if summaries:
                rendered_parts.append(
                    "According to the Department of Agriculture guidelines: " + " ".join(summaries)
                )

        final_text = "\n\n".join(rendered_parts) if rendered_parts else "No specific evidence was collected for this query."
        return final_text, ui_components

    @staticmethod
    def _summarize_snippets(snippets: Any, limit: int = 2) -> List[str]:
        """Summarize retrieved chunks, dropping near-duplicates from overlapping windows."""
        if not snippets:
            return []

        seen = set()
        out: List[str] = []
        for snip in snippets:
            raw = snip.get("content", "") if isinstance(snip, dict) else snip
            summary = summarize_snippet(raw)
            if not summary:
                continue
            key = summary.lower().rstrip(".")
            if key in seen:
                continue
            seen.add(key)
            out.append(summary)
            if len(out) >= limit:
                break
        return out
