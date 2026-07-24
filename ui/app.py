"""
Multi-Agent Paddy Disease Diagnostic & Fertilizer Recommendation System
Farmer-Centric Web Application (ui/app.py)

Module: IT41043 - Agentic AI
Author: Chamod

Modern UI/UX Architecture:
  1. Hero Banner + Quick Chips Grid
  2. Above-The-Fold Chat Input Box (No Auto-Scroll down)
  3. Conversational Message History Thread & Advisory Tabs
  4. Perfectly Aligned 3-Column Farmer Toolkit Grid
"""

import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

# ── Page Config ──
st.set_page_config(
    page_title="PaddyAgri AI - Sri Lankan Agriculture Smart Portal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Modern UI CSS System ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0b291a 0%, #134027 50%, #1e5a37 100%);
        color: #ffffff;
        padding: 2.0rem 2.0rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(64, 192, 87, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.0rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff, #80ed99);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #c7f9cc;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* Glassmorphic Cards inside containers */
    .glass-card {
        background: rgba(20, 46, 32, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(64, 192, 87, 0.22);
        border-radius: 14px;
        padding: 1.0rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 0.8rem;
    }
    .glass-card-title {
        font-weight: 700;
        font-size: 1.0rem;
        color: #80ed99;
        margin-bottom: 0.4rem;
    }

    /* Badges */
    .badge-pass {
        background: rgba(40, 167, 69, 0.2);
        color: #75b798;
        padding: 0.6rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid rgba(40, 167, 69, 0.4);
        margin-bottom: 0.8rem;
    }
    .badge-warn {
        background: rgba(255, 193, 7, 0.2);
        color: #ffda6a;
        padding: 0.6rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid rgba(255, 193, 7, 0.4);
        margin-bottom: 0.8rem;
    }

    /* Disease Cards */
    .disease-item {
        background: rgba(15, 34, 23, 0.7);
        border: 1px solid rgba(64, 192, 87, 0.18);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
    }
    .disease-name {
        font-weight: 700;
        color: #57cc99;
        font-size: 0.95rem;
    }
    .disease-desc {
        font-size: 0.85rem;
        color: #c7f9cc;
        margin-top: 0.2rem;
        line-height: 1.4;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.title("🌾 PaddyAgri AI")
    st.caption("Sri Lankan Agriculture Smart Portal")

    st.divider()

    # System Status
    st.subheader("⚙️ System Status")
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if gemini_key:
        st.success("🟢 Google Gemini Ready")
    if groq_key:
        st.success("🟢 Groq Llama 3.3 70B Ready")
    if cohere_key:
        st.success("🟢 Cohere Command R+ Ready")
    if openrouter_key:
        st.info("🔵 OpenRouter Standby")
    if not (gemini_key or groq_key or cohere_key or openrouter_key):
        st.error("🔴 API Key Missing")
        st.info("Set `GEMINI_API_KEY`, `GROQ_API_KEY`, or `COHERE_API_KEY` in `.env`.")

    st.divider()

    # Advanced AI Developer Mode Toggle
    dev_mode = st.checkbox(
        "⚡ Advanced AI Developer Mode (Raw Agent Communication Trace)",
        value=False
    )
    if dev_mode:
        st.warning("Developer Mode ON — raw Agent-to-Agent JSON traces visible in assistant response tabs.")

    st.divider()

    # 📞 Agrarian Helpline
    st.markdown("### 📞 Agrarian Helpline")
    st.info(
        "**Department of Agriculture — 1920**\n\n"
        "☎️ **1920** (Govi Sahana Piyasa)\n\n"
        "🌐 [doa.gov.lk](http://www.doa.gov.lk)\n\n"
        "*Free expert advice for Sri Lankan farmers.*"
    )

    st.divider()

    # 🗑️ Clear Chat History
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if "pending_chip_query" in st.session_state:
            st.session_state.pop("pending_chip_query", None)
        st.rerun()

    st.markdown(
        "🔗 [DOA Advisory PDF Corpus](https://drive.google.com/drive/folders/1O6Teo6_gPBZOd27rtzAI84RSTKKU5er9?usp=sharing)"
    )


# ══════════════════════════════════════════════
# SECTION 1 — HERO BANNER & QUICK SUGGESTIONS
# ══════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌾 PaddyAgri AI - Sri Lankan Agriculture Smart Portal</div>
    <div class="hero-sub">Smart Paddy Disease Diagnosis & Fertilizer Recommendations for Sri Lankan Farmers</div>
</div>
""", unsafe_allow_html=True)

# ── Quick Suggestion Chips ──
st.markdown("##### 💡 Quick Suggestions — click any chip to auto-submit:")
chip1, chip2, chip3, chip4 = st.columns(4)

quick_queries = {
    "chip1": "What are the common symptoms of Paddy Blast disease and what chemical and organic treatments does the DOA recommend?",
    "chip2": "What are the recommended Urea, TSP, and MOP fertilizer rates per acre for the Yala crop season in Sri Lanka?",
    "chip3": "How do I identify and control Brown Planthopper (BPH) pest attacks in paddy fields using DOA certified methods?",
    "chip4": "What quality standards, purity percentages, and germination rates are required for Certified Seed Paddy in Sri Lanka?"
}

if chip1.button("🍂 Paddy Blast Symptoms & Treatments", use_container_width=True):
    st.session_state["pending_chip_query"] = quick_queries["chip1"]
    st.rerun()

if chip2.button("🌱 Yala Season NPK Fertilizer Rates", use_container_width=True):
    st.session_state["pending_chip_query"] = quick_queries["chip2"]
    st.rerun()

if chip3.button("🐛 Brown Planthopper (BPH) Control", use_container_width=True):
    st.session_state["pending_chip_query"] = quick_queries["chip3"]
    st.rerun()

if chip4.button("📜 Certified Seed Paddy Standards", use_container_width=True):
    st.session_state["pending_chip_query"] = quick_queries["chip4"]
    st.rerun()

st.divider()


# ══════════════════════════════════════════════
# SECTION 2 — NATIVE CHAT INPUT (Placed Top / Above-the-Fold)
# ══════════════════════════════════════════════

user_query = st.chat_input(
    "Ask your paddy farming question in English, Sinhala, or Singlish...",
    key="chat_input_main"
)

# Helper function to run orchestrator
def run_orchestrator(query: str):
    from core.agent_orchestrator import PaddyAgentOrchestrator
    orchestrator = PaddyAgentOrchestrator()
    return orchestrator.process_user_request(query)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle pending chip query
pending = st.session_state.pop("pending_chip_query", None)
if pending:
    with st.chat_message("user"):
        st.markdown(pending)
    with st.spinner("🔄 Agents searching agricultural handbooks in parallel and checking safety rules…"):
        try:
            response = run_orchestrator(pending)
        except Exception as e:
            st.error(f"Runtime Error: {e}")
            st.info("Check your `.env` API keys and agent orchestrator imports.")
            st.stop()

    st.session_state.messages.append({"role": "user", "content": pending})
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.final_synthesis,
        "response_obj": response,
        "query": pending
    })
    st.rerun()

# Handle new user chat input
if user_query and user_query.strip():
    st.session_state.messages.append({"role": "user", "content": user_query.strip()})
    with st.chat_message("user"):
        st.markdown(user_query.strip())

    with st.spinner("🔄 Agents searching agricultural handbooks in parallel and checking safety rules…"):
        try:
            response = run_orchestrator(user_query.strip())
        except Exception as e:
            st.error(f"Runtime Error: {e}")
            st.info("Check your `.env` API keys and agent orchestrator imports.")
            st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.final_synthesis,
        "response_obj": response,
        "query": user_query.strip()
    })
    st.rerun()


# ══════════════════════════════════════════════
# SECTION 3 — CHAT HISTORY MESSAGES & ADVISORY TABS
# ══════════════════════════════════════════════

if st.session_state.messages:
    st.markdown("### 💬 Advisory Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🌾"):
            response = msg.get("response_obj")
            if response is None:
                st.markdown(msg["content"])
                continue

            # Dynamic Tab Labels based on language
            msg_query = msg.get("query", msg.get("content", ""))
            from config.model_provider import detect_language_and_script
            is_sinhala = detect_language_and_script(msg_query)

            if is_sinhala:
                tab_labels = [
                    "📋 ගොවි උපදෙස් පත්‍රිකාව",
                    "📚 තහවුරු කළ රාජ්‍ය මූලාශ්‍ර",
                    "🛡️ ආරක්‍ෂිත සහ රාජ්‍ය ප්‍රමිතීන්"
                ]
            else:
                tab_labels = [
                    "📋 Advisory Report",
                    "📚 Official DOA Guidelines",
                    "🛡️ Safety & Regulatory Checks"
                ]

            if dev_mode:
                tab_labels.append("⚡ Developer Agent Trace")

            tabs = st.tabs(tab_labels)

            # TAB 1 — Formatted Advisory Report & Download TXT
            with tabs[0]:
                st.markdown(response.final_synthesis)
                st.download_button(
                    label="📥 Download Advisory Report (.txt)",
                    data=response.final_synthesis,
                    file_name="paddy_advisory_report.txt",
                    mime="text/plain",
                    key=f"dl_{id(response)}"
                )

            # TAB 2 — DOA PDF Citations
            with tabs[1]:
                st.markdown("### 📚 Official DOA Handbook Citations")
                all_sources = []
                if hasattr(response, "diagnostic_info") and response.diagnostic_info:
                    all_sources.extend(response.diagnostic_info.rag_sources)
                if hasattr(response, "fertilizer_info") and response.fertilizer_info:
                    all_sources.extend(response.fertilizer_info.rag_sources)

                if all_sources:
                    for i, src in enumerate(all_sources, 1):
                        with st.expander(f"📄 Source #{i} — {src.filename} · Page {src.page}"):
                            st.write(
                                f"**Category:** `{src.category}` · "
                                f"**Document:** `{src.filename}` · "
                                f"**Page:** {src.page}"
                            )
                            st.markdown(f'> *"{src.content}"*')
                else:
                    st.info("No direct handbook citations found for this query. Advisory compiled from general DOA guidelines.")

            # TAB 3 — Safety & Regulatory Compliance
            with tabs[2]:
                st.markdown("### 🛡️ Pesticide Act & Fertilizer Ordinance Compliance")
                refl = response.reflection_result if hasattr(response, "reflection_result") else None
                if refl:
                    if refl.all_checks_passed:
                        st.markdown(
                            '<div class="badge-pass">'
                            "✅ ALL CHECKS PASSED — Recommendations comply with Pesticide Act No.33 and Fertilizer Ordinance limits"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="badge-warn">'
                            "⚠️ ATTENTION — Some chemicals may be restricted or require additional precautions"
                            "</div>",
                            unsafe_allow_html=True
                        )

                    st.markdown("#### Safety Verdict Details:")
                    for v in refl.verdicts:
                        icon = "✅" if v.passed else "⚠️"
                        st.write(f"{icon} **{v.check_name}** — {v.message}")

                    if refl.regulatory_citations:
                        st.markdown("#### Regulatory References:")
                        for cite in refl.regulatory_citations:
                            st.write(f"📜 {cite}")
                else:
                    st.info("General query — no chemical safety or regulatory checks required.")

            # TAB 4 — Developer JSON Trace (only when toggled)
            if dev_mode:
                with tabs[3]:
                    st.markdown("### ⚡ Developer Diagnostics — Agent-to-Agent Message Trace")
                    intent_val = response.intent.value if hasattr(response, "intent") else "N/A"
                    trace_len = len(response.message_trace) if hasattr(response, "message_trace") else 0
                    st.info(f"**Intent:** {intent_val} · **Messages Exchanged:** {trace_len}")
                    if hasattr(response, "message_trace") and response.message_trace:
                        for idx, mt in enumerate(response.message_trace, 1):
                            with st.expander(
                                f"✉️ #{idx}: {mt.sender} ➔ {mt.receiver} [{mt.intent.value}]",
                                expanded=True
                            ):
                                st.json({
                                    "message_id": mt.message_id,
                                    "sender": mt.sender,
                                    "receiver": mt.receiver,
                                    "intent": mt.intent.value,
                                    "timestamp": mt.timestamp,
                                    "payload": mt.payload
                                })


# ══════════════════════════════════════════════
# SECTION 4 — PERFECTLY ALIGNED RESPONSIVE GRID
# ══════════════════════════════════════════════

st.divider()
st.markdown("## 🧰 Farmer Toolkit")

tool_col1, tool_col2, tool_col3 = st.columns(3)

# ── Column 1: Fertilizer Calculator ──
with tool_col1:
    with st.container(border=True):
        st.markdown("### 💰 Fertilizer Calculator")
        st.caption("Estimate bags & cost per acre")

        acres = st.number_input(
            "Field size (Acres):", min_value=0.5, max_value=50.0,
            value=2.0, step=0.5, key="calc_acres"
        )
        calc_season = st.selectbox(
            "Crop season:", ["Yala (Dry)", "Maha (Wet)"], key="calc_season"
        )

        if "Yala" in calc_season:
            urea_bags = round((acres * 50) / 50, 1)
            tsp_bags  = round((acres * 25) / 50, 1)
            mop_bags  = round((acres * 25) / 50, 1)
        else:
            urea_bags = round((acres * 65) / 50, 1)
            tsp_bags  = round((acres * 30) / 50, 1)
            mop_bags  = round((acres * 30) / 50, 1)

        cost_lkr = (urea_bags + tsp_bags + mop_bags) * 2500

        st.write(f"• **Urea (50 kg bags):** {urea_bags}")
        st.write(f"• **TSP (50 kg bags):** {tsp_bags}")
        st.write(f"• **MOP (50 kg bags):** {mop_bags}")
        st.success(f"**Estimated Cost:** LKR {cost_lkr:,.2f}")

# ── Column 2: Seasonal Advisory ──
with tool_col2:
    with st.container(border=True):
        st.markdown("### 🗓️ Seasonal Advisory")
        st.caption("Best practices for Yala & Maha")

        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">🌾 Yala Season (May – August)</div>
            <ul style="margin:0; padding-left:1.2rem;">
                <li>Harvest standing crops promptly to avoid rain.</li>
                <li>Practice dry tillage to break hardpan.</li>
                <li>Apply basal fertilizer at land prep.</li>
            </ul>
        </div>
        <div class="glass-card" style="margin-bottom:0;">
            <div class="glass-card-title">🌧️ Maha Season (Sept – March)</div>
            <ul style="margin:0; padding-left:1.2rem;">
                <li>Incorporate organic matter during wet ploughing.</li>
                <li>Select DOA-certified seed varieties.</li>
                <li>Monitor water levels — avoid flooding.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ── Column 3: Paddy Disease Guide ──
with tool_col3:
    with st.container(border=True):
        st.markdown("### 🖼️ Paddy Disease Guide")
        st.caption("Visual symptoms & fast diagnosis")

        st.markdown("""
        <div class="disease-item">
            <div class="disease-name">🍂 Paddy Blast (Pyricularia oryzae)</div>
            <div class="disease-desc">Diamond-shaped spots. Spray Tricyclazole.</div>
        </div>
        <div class="disease-item">
            <div class="disease-name">🟤 Brown Spot (Bipolaris oryzae)</div>
            <div class="disease-desc">Oval brown leaf lesions. Apply K & Zn.</div>
        </div>
        <div class="disease-item">
            <div class="disease-name">🌿 Sheath Blight (Rhizoctonia)</div>
            <div class="disease-desc">Grey-green patches. Apply Hexaconazole.</div>
        </div>
        <div class="disease-item" style="margin-bottom:0;">
            <div class="disease-name">🐛 Brown Planthopper (BPH)</div>
            <div class="disease-desc">Circular yellowing ('hopper burn'). Drain fields.</div>
        </div>
        """, unsafe_allow_html=True)
