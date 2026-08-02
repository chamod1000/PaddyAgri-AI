"""
Response Experience Engine - PaddyAgri-AI (CACAA-AO Architecture)
Author: Lead Conversation Designer & Senior AI UX Architect

Translates raw agent evidence artifacts into dynamic, conversational,
ChatGPT/Claude-quality responses customized by query intent.

Supported Intent Formats:
  1. KNOWLEDGE: Overview -> Common Symptoms -> Chemical Treatment -> Organic Practices -> Prevention -> Sources
  2. DIAGNOSIS: Initial Assessment -> Most Likely Disease -> Confidence -> Why Diagnosis -> Immediate Actions -> Recommended Treatment -> Prevention -> Sources
  3. IMAGE DIAGNOSIS: Image Analysis -> Observed Symptoms -> Most Likely Disease -> Confidence -> Treatment -> Next Steps -> Sources
  4. WEATHER: Weather Summary -> Disease Risk -> Field Recommendation -> Fertilizer Advice -> Sources
  5. FERTILIZER: Fertilizer Recommendation -> Growth Stage & Timetable -> NPK Dosage -> Application Schedule -> Important Notes -> Sources
  6. MIXED: Unified Natural Narrative combining Diagnosis + Weather + Fertilizer seamlessly into 1 answer.
"""

import re
from typing import Dict, Any, Optional, List, Generator
from config.model_provider import get_reasoning_model


class ResponseExperienceEngine:
    """Intelligent Conversational UX Engine for PaddyAgri-AI."""

    @classmethod
    def get_val(cls, obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def detect_experience_mode(
        cls,
        user_query: str,
        diagnostic_info: Optional[Any] = None,
        fertilizer_info: Optional[Any] = None,
        weather_info: Optional[Any] = None,
        vision_info: Optional[Any] = None,
        general_info: Optional[Any] = None
    ) -> str:
        """Determines the exact conversational experience layout mode."""
        q_lower = user_query.lower()

        count_domains = 0
        if diagnostic_info and cls.get_val(diagnostic_info, "suspected_disease"): count_domains += 1
        if fertilizer_info and (cls.get_val(fertilizer_info, "urea_dosage_per_acre_kg") or cls.get_val(fertilizer_info, "urea_kg")): count_domains += 1
        if weather_info and cls.get_val(weather_info, "location"): count_domains += 1

        if count_domains >= 2:
            return "MIXED"

        if vision_info or any(kw in q_lower for kw in ["image", "photo", "uploaded", "picture", "leaf photo"]):
            return "IMAGE_DIAGNOSIS"

        if diagnostic_info and cls.get_val(diagnostic_info, "suspected_disease"):
            # Check if informational definition vs active symptom diagnosis
            is_info_query = any(q_lower.startswith(w) for w in ["what is", "explain", "tell me about", "define", "how does"])
            if is_info_query and not any(kw in q_lower for kw in ["my rice", "my paddy", "my leaves", "my field", "affected", "lesions"]):
                return "KNOWLEDGE"
            return "DIAGNOSIS"

        if weather_info and cls.get_val(weather_info, "location"):
            return "WEATHER"

        if fertilizer_info and (cls.get_val(fertilizer_info, "urea_dosage_per_acre_kg") or cls.get_val(fertilizer_info, "urea_kg")):
            return "FERTILIZER"

        return "KNOWLEDGE"

    @classmethod
    def generate_conversational_response(
        cls,
        user_query: str,
        diagnostic_info: Optional[Any] = None,
        fertilizer_info: Optional[Any] = None,
        weather_info: Optional[Any] = None,
        vision_info: Optional[Any] = None,
        general_info: Optional[Any] = None,
        reflection_result: Optional[Any] = None,
        final_synthesis: Optional[str] = None
    ) -> str:
        """
        Generates a beautifully structured, conversational, ChatGPT/Claude-quality response.
        """
        mode = cls.detect_experience_mode(
            user_query=user_query,
            diagnostic_info=diagnostic_info,
            fertilizer_info=fertilizer_info,
            weather_info=weather_info,
            vision_info=vision_info,
            general_info=general_info
        )

        # Build Context String for LLM
        ctx_parts = []
        if weather_info:
            loc = cls.get_val(weather_info, "location", "Anuradhapura")
            temp = cls.get_val(weather_info, "temperature_c", 31.5)
            hum = cls.get_val(weather_info, "humidity_pct", 82)
            alert = cls.get_val(weather_info, "fungal_risk_alert", "High Risk")
            notes = cls.get_val(weather_info, "advisory_notes", [])
            ctx_parts.append(f"Weather Context: Location={loc}, Temp={temp}°C, Humidity={hum}%, Risk={alert}, Advice={notes}")

        if diagnostic_info:
            dis = cls.get_val(diagnostic_info, "suspected_disease", "Paddy Disease")
            conf = cls.get_val(diagnostic_info, "confidence_level") or cls.get_val(diagnostic_info, "confidence", "High")
            sym = cls.get_val(diagnostic_info, "symptoms_identified", [])
            trt = cls.get_val(diagnostic_info, "treatment_recommended", [])
            ctx_parts.append(f"Diagnostic Context: Disease={dis}, Confidence={conf}, Symptoms={sym}, Treatments={trt}")

        if fertilizer_info:
            season = cls.get_val(fertilizer_info, "season", "Yala/Maha")
            urea = cls.get_val(fertilizer_info, "urea_dosage_per_acre_kg") or cls.get_val(fertilizer_info, "urea_kg", 50.0)
            tsp = cls.get_val(fertilizer_info, "tsp_dosage_per_acre_kg") or cls.get_val(fertilizer_info, "tsp_kg", 25.0)
            mop = cls.get_val(fertilizer_info, "mop_dosage_per_acre_kg") or cls.get_val(fertilizer_info, "mop_kg", 25.0)
            sched = cls.get_val(fertilizer_info, "application_schedule", []) or cls.get_val(fertilizer_info, "notes", [])
            ctx_parts.append(f"Fertilizer Context: Season={season}, Urea={urea}kg/acre, TSP={tsp}kg/acre, MOP={mop}kg/acre, Schedule={sched}")

        if general_info and isinstance(general_info, dict):
            snippets = general_info.get("snippets", []) or general_info.get("chunks", [])
            for snip in snippets[:3]:
                ctx_parts.append(f"DOA Knowledge Context: {snip.get('content', '').strip()[:300]}")

        full_context = "\n".join(ctx_parts)

        # Custom Prompting according to experience mode
        prompt = cls._build_prompt_for_mode(mode, user_query, full_context)

        # Execute LLM Reasoning Engine
        llm_response = None
        try:
            model = get_reasoning_model()
            resp = model.invoke(prompt)
            if resp and getattr(resp, "content", None):
                llm_response = resp.content.strip()
        except Exception as e:
            print(f"[RESPONSE ENGINE WARNING] LLM generation fallback: {e}")

        if not llm_response:
            llm_response = cls._fallback_response(mode, user_query, diagnostic_info, fertilizer_info, weather_info, general_info)

        # Ensure Sources are formatted neatly at the end
        sources_section = cls._build_sources_section(general_info)
        if sources_section and "📚 Sources" not in llm_response:
            llm_response += f"\n\n{sources_section}"

        return llm_response

    @classmethod
    def _build_prompt_for_mode(cls, mode: str, user_query: str, context: str) -> str:
        base_head = """You are PaddyAgri AI, an experienced, friendly, and highly knowledgeable agricultural expert helping a Sri Lankan paddy farmer.
Your tone must be warm, professional, farmer-friendly, clear, and easy to read. Speak directly to the farmer.
Never sound like software. Avoid technical jargon like 'Diagnostic Advisory', 'Confidence Level', 'Key Symptoms Identified', 'JSON', or 'Agents'.
Always start with a direct answer or friendly overview.
Ground all factual recommendations in the provided Department of Agriculture (DOA) context."""

        if mode == "KNOWLEDGE":
            return f"""{base_head}

User Query: {user_query}

Verified Context:
{context}

Format your response using ONLY the following section structure:
🌾 Overview (Direct, friendly answer)
🔍 Symptoms (Common signs to watch for)
🧪 Chemical Treatment (DOA recommended fungicides/pesticides)
🌱 Organic Practices (Organic & cultural management)
🛡 Prevention (Long-term preventive tips)

Important: Do NOT include Confidence levels. Do NOT mention internal code names."""

        elif mode == "DIAGNOSIS":
            return f"""{base_head}

User Query: {user_query}

Verified Context:
{context}

Format your response using ONLY the following section structure:
🌾 Initial Assessment (Empathetic, direct assessment based on described symptoms)
🦠 Most Likely Disease (Identify the specific paddy disease)
📊 Confidence (Display confidence, e.g. High / Medium)
🔍 Why this diagnosis? (Briefly explain why the symptoms point to this disease)
⚠ Immediate Actions (Urgent steps the farmer should take right now)
🧪 Recommended Treatment (DOA chemical treatments)
🌱 Organic Practices (Organic alternatives)
🛡 Prevention (Steps to prevent future outbreaks)"""

        elif mode == "IMAGE_DIAGNOSIS":
            return f"""{base_head}

User Query: {user_query}

Verified Context:
{context}

Format your response using ONLY the following section structure:
🖼 Image Analysis (Visual assessment of the uploaded leaf photo)
🔍 Observed Symptoms (Key visual features seen on the leaf)
🦠 Most Likely Disease (Disease identification)
📊 Confidence (Diagnostic confidence level)
🧪 Treatment (DOA recommended chemical and organic treatments)
🛡 Next Steps (Field management recommendations)"""

        elif mode == "WEATHER":
            return f"""{base_head}

User Query: {user_query}

Verified Context:
{context}

Format your response using ONLY the following section structure:
🌦 Weather Summary (Temperature, humidity, and atmospheric conditions)
🦠 Disease Risk (Fungal & bacterial risk assessment)
📋 Field Recommendation (Field operations advice)
🌱 Fertilizer Advice (Specific advice on fertilizer timing relative to rain/humidity)"""

        elif mode == "FERTILIZER":
            return f"""{base_head}

User Query: {user_query}

Verified Context:
{context}

Format your response using ONLY the following section structure:
🌱 Fertilizer Recommendation (Direct, clear guidance)
📅 Growth Stage & Timetable (Season and growth stage breakdown)
⚖️ NPK Recommendation (Urea, TSP, MOP dosage per acre)
📋 Application Schedule (Splits and application timing)
⚠ Important Notes (Field moisture and safety rules)"""

        else:  # MIXED
            return f"""{base_head}

User Query: {user_query}

Verified Context:
{context}

The user asked a multi-part query combining diagnosis, weather, or fertilizer.
Integrate all elements smoothly into ONE cohesive, natural, conversational response.
Never split into separate artificial reports. Use natural headings like 🌾 Initial Assessment, 🌦 Weather Impact, 🧪 Treatment & Fertilizer Guidance, and 🛡 Prevention."""

    @classmethod
    def _build_sources_section(cls, general_info: Optional[Any]) -> str:
        if not general_info or not isinstance(general_info, dict):
            return ""

        snippets = general_info.get("snippets", []) or general_info.get("chunks", [])
        if not snippets:
            return ""

        lines = ["### 📚 Sources"]
        seen = set()
        for snip in snippets[:3]:
            fn = snip.get("filename", "DOA Technical Guide")
            pg = snip.get("page", 1)
            cat = snip.get("category", "Department of Agriculture")
            
            # Map filename to clean user-friendly publication name
            pub_name = "Department of Agriculture Sri Lanka (DOA Manual)"
            if "Rice-Congress" in fn:
                pub_name = "Rice Congress 2010 Research Publication"
            elif "ROP" in fn:
                pub_name = "DOA Rice Operations & Pathology Guide"
            elif "Danapala" in fn:
                pub_name = "Rice Pathology Research (Dr. M.P. Dhanapala)"
            elif "learned" in fn:
                pub_name = "DOA Verified Pathology Records"

            item_key = f"{pub_name} (Page {pg})"
            if item_key not in seen:
                seen.add(item_key)
                lines.append(f"• {pub_name} (Page {pg})")

        return "\n".join(lines) if len(lines) > 1 else ""

    @classmethod
    def _fallback_response(
        cls,
        mode: str,
        user_query: str,
        diagnostic_info: Optional[Any],
        fertilizer_info: Optional[Any],
        weather_info: Optional[Any],
        general_info: Optional[Any]
    ) -> str:
        """Deterministic local template fallback when LLM is unavailable."""
        out = []

        if mode == "KNOWLEDGE":
            out.append("### 🌾 Overview")
            out.append("Based on Department of Agriculture research, here is the essential guide for this paddy crop query.\n")
            if diagnostic_info:
                dis = cls.get_val(diagnostic_info, "suspected_disease", "Paddy Disease")
                out.append(f"**Disease:** {dis}\n")
                syms = cls.get_val(diagnostic_info, "symptoms_identified", [])
                if syms:
                    out.append("### 🔍 Common Symptoms")
                    for s in syms: out.append(f"- {s}")
                    out.append("")
                trts = cls.get_val(diagnostic_info, "treatment_recommended", [])
                if trts:
                    out.append("### 🧪 Chemical Treatment")
                    for t in trts[:2]: out.append(f"- {t}")
                    out.append("\n### 🌱 Organic Practices")
                    for t in trts[2:]: out.append(f"- {t}")
                    out.append("")

        elif mode == "DIAGNOSIS":
            dis = cls.get_val(diagnostic_info, "suspected_disease", "Paddy Blast")
            conf = cls.get_val(diagnostic_info, "confidence_level", "High")
            out.append("### 🌾 Initial Assessment")
            out.append("Based on the symptoms you described, your paddy crop appears to be affected by pathology stress.\n")
            out.append(f"### 🦠 Most Likely Disease\n**{dis}**\n")
            out.append(f"### 📊 Confidence\n`{conf}`\n")

            syms = cls.get_val(diagnostic_info, "symptoms_identified", [])
            if syms:
                out.append("### 🔍 Why this diagnosis?")
                for s in syms: out.append(f"- {s}")
                out.append("")

            trts = cls.get_val(diagnostic_info, "treatment_recommended", [])
            if trts:
                out.append("### ⚠ Immediate Actions")
                out.append("- Drain excess standing water if humidity is high to stop fungal spore spread.")
                out.append("\n### 🧪 Recommended Treatment")
                for t in trts: out.append(f"- {t}")
                out.append("")

        elif mode == "WEATHER":
            loc = cls.get_val(weather_info, "location", "Anuradhapura")
            temp = cls.get_val(weather_info, "temperature_c", 31.5)
            hum = cls.get_val(weather_info, "humidity_pct", 82)
            alert = cls.get_val(weather_info, "fungal_risk_alert", "High Risk")
            out.append(f"### 🌦 Weather Summary ({loc})")
            out.append(f"- **Temperature:** `{temp}°C`\n- **Relative Humidity:** `{hum}%`\n")
            out.append(f"### 🦠 Disease Risk\n**Fungal Disease Risk:** `{alert}`\n")
            notes = cls.get_val(weather_info, "advisory_notes", [])
            if notes:
                out.append("### 📋 Field Recommendation")
                for n in notes: out.append(f"- {n}")
                out.append("")

        elif mode == "FERTILIZER":
            season = cls.get_val(fertilizer_info, "season", "Yala/Maha")
            urea = cls.get_val(fertilizer_info, "urea_dosage_per_acre_kg") or cls.get_val(fertilizer_info, "urea_kg", 50.0)
            tsp = cls.get_val(fertilizer_info, "tsp_dosage_per_acre_kg") or cls.get_val(fertilizer_info, "tsp_kg", 25.0)
            mop = cls.get_val(fertilizer_info, "mop_dosage_per_acre_kg") or cls.get_val(fertilizer_info, "mop_kg", 25.0)
            out.append(f"### 🌱 Fertilizer Recommendation ({season} Season)")
            out.append(f"Here are the Department of Agriculture NPK rates per acre:\n")
            out.append("### ⚖️ NPK Recommendation")
            out.append(f"- **Urea (46% N):** `{urea:.1f} kg/acre`")
            out.append(f"- **TSP (46% P₂O₅):** `{tsp:.1f} kg/acre`")
            out.append(f"- **MOP (60% K₂O):** `{mop:.1f} kg/acre`\n")
            sched = cls.get_val(fertilizer_info, "application_schedule", []) or cls.get_val(fertilizer_info, "notes", [])
            if sched:
                out.append("### 📋 Application Schedule")
                for s in sched: out.append(f"- {s}")
                out.append("")

        else:
            out.append("### 🌾 Advisory Overview")
            out.append("Here is the agricultural advisory for your query based on Department of Agriculture recommendations.")

        return "\n".join(out)
