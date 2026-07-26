"""
Multi-Agent Paddy Disease Diagnostic & Fertilizer Recommendation System
Farmer-Centric Web Application (ui/app.py)

Module: IT41043 - Agentic AI
Author: Chamod

Modern UI/UX Architecture:
  1. Sprout Paddy Emerald Theme + WCAG AAA High Contrast Readability
  2. Above-The-Fold Chat Input Box (Zero Auto-Scroll Jump on Initial Load)
  3. Conversational Message History & 5-Section Detailed Advisory Reports
  4. Perfectly Aligned 3-Column Glassmorphic Farmer Toolkit Grid
"""

import os
import sys
import json
import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv

nest_asyncio.apply()

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

# ── Advanced Custom CSS Injection (Glassmorphism, High-Contrast & Micro-Interactions) ──
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* CSS Root Variable Tokenization */
    :root {
        --bg-primary: #06120b;
        --bg-surface: rgba(14, 36, 24, 0.85);
        --emerald-primary: #40c057;
        --emerald-glow: #80ed99;
        --harvest-gold: #fcc419;
        --text-primary: #f4fce3;
        --text-secondary: #b7e4c7;
        --border-glass: rgba(64, 192, 87, 0.28);
    }

    /* Base Environment */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        scroll-behavior: smooth !important;
    }

    /* Restore Streamlit Material Icons font rendering */
    button[data-testid="stSidebarCollapseButton"] span,
    button[data-testid="stSidebarCollapseButton"] i,
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-icons {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    }

    /* Hide standard streamlit elements */
    footer {visibility: hidden;}

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #092215 0%, #113821 50%, #1b5231 100%);
        color: #ffffff;
        padding: 2.0rem 2.0rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid var(--border-glass);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff, var(--emerald-glow));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* Clean Flat Executive Advisory Card */
    .advisory-card-flat {
        background: rgba(14, 36, 24, 0.5) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        padding: 1.4rem !important;
        box-shadow: none !important;
        margin-bottom: 1.0rem !important;
    }

    /* Glassmorphic Container Cards */
    .glass-card {
        background: var(--bg-surface);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 0.8rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card-hoverable:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(64, 192, 87, 0.3);
    }
    .glass-card-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--emerald-glow);
        margin-bottom: 0.4rem;
    }

    /* Animated LED Indicator Pills */
    .led-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(6, 18, 11, 0.7);
        border: 1px solid var(--border-glass);
        padding: 7px 14px;
        border-radius: 50px;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 8px;
        width: 100%;
    }
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 10px;
        background-color: var(--emerald-primary);
        box-shadow: 0 0 10px var(--emerald-primary), 0 0 20px var(--emerald-primary);
        animation: statusPulse 2s infinite;
    }
    @keyframes statusPulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(64, 192, 87, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(64, 192, 87, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(64, 192, 87, 0); }
    }

    /* Action Buttons */
    .stButton button {
        background: rgba(17, 48, 30, 0.8) !important;
        border: 1px solid var(--border-glass) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background: #194d2e !important;
        border-color: var(--emerald-glow) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 18px rgba(64, 192, 87, 0.35) !important;
        color: #ffffff !important;
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
# SIDEBAR (No Header Image - Clean Text Titles)
# ══════════════════════════════════════════════
with st.sidebar:
    st.title("🌾 PaddyAgri AI")
    st.caption("Sri Lankan Agriculture Smart Portal")

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
# SECTION 2 — TOP-POSITIONED MODERN QUERY INPUT (UI/UX Best Practice)
# ══════════════════════════════════════════════

with st.form(key="top_query_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query_text = st.text_input(
            "Ask Agri AI Question",
            placeholder="✨ Type your agricultural query, disease symptoms, or fertilizer question...",
            label_visibility="collapsed",
            key="top_search_input"
        )
    with col_btn:
        submit_clicked = st.form_submit_button("⚡ Ask AI Assistant", use_container_width=True)

user_query = query_text.strip() if (submit_clicked and query_text and query_text.strip()) else None

# Helper function to run orchestrator with live step highlights
def run_orchestrator_with_live_highlights(query: str, status):
    step_box = st.empty()

    def render_steps(active_step: int, live_msg: str = "", done: bool = False):
        steps = [
            ("🎯", "1. RouterAgent", "Classifying query intent & agricultural domain context (OpenRouter / Groq)"),
            ("📚", "2. RAG Retriever", "Searching 20+ DOA PDF Handbooks in parallel (FAISS Index)"),
            ("🔬", "3. Diagnostic & Fertilizer Agents", "Synthesizing pathology & NPK schedule (Claude / Gemini Pro / Llama 70B)"),
            ("🛡️", "4. Regulatory Compliance", "Pesticide Act No. 33 & DOA Fertilizer Ordinance Verified (Built-in Filter)")
        ]
        html_content = '<div style="margin-top: 6px;">'
        for idx, (icon, title, desc) in enumerate(steps, 1):
            if done or idx < active_step:
                html_content += f'''
                <div style="padding: 8px 14px; margin-bottom: 8px; border-radius: 10px; background: rgba(64, 192, 87, 0.12); border: 1px solid rgba(64, 192, 87, 0.4); color: #80ed99; font-size: 0.93rem;">
                    ✅ <b>{title}</b> — <span style="color: #b7e4c7;">{desc}</span> <span style="float: right; font-weight: 700; color: #40c057;">[COMPLETED]</span>
                </div>'''
            elif idx == active_step:
                detail_text = f" — <span style='color: #fff;'>{live_msg or desc}</span>"
                html_content += f'''
                <div style="padding: 10px 15px; margin-bottom: 8px; border-radius: 10px; background: linear-gradient(90deg, rgba(64, 192, 87, 0.25), rgba(252, 196, 25, 0.2)); border: 2px solid #fcc419; color: #ffffff; font-weight: 600; font-size: 0.95rem; box-shadow: 0 0 15px rgba(252, 196, 25, 0.4);">
                    ⏳ {icon} <b>{title}</b>{detail_text} <span style="float: right; color: #fcc419; font-weight: 800;">⚡ EXECUTING...</span>
                </div>'''
            else:
                html_content += f'''
                <div style="padding: 8px 14px; margin-bottom: 8px; border-radius: 10px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); color: #7a8880; font-size: 0.9rem;">
                    ⏸️ {icon} <b>{title}</b> — <span>{desc}</span> <span style="float: right; color: #6c757d;">[WAITING]</span>
                </div>'''
        html_content += '</div>'
        step_box.markdown(html_content, unsafe_allow_html=True)

    render_steps(active_step=1, live_msg="Initializing Swarm Agents...")

    def live_step_callback(step_num: int, label: str):
        status.update(label=f"⚡ Live Swarm Tracking — Step {step_num}/4: {label}", state="running")
        render_steps(active_step=step_num, live_msg=label)

    from core.agent_orchestrator import PaddyAgentOrchestrator
    orchestrator = PaddyAgentOrchestrator()
    response, synthesis_agent = orchestrator.process_user_request(query, stream=True, step_callback=live_step_callback)

    render_steps(active_step=4, done=True)
    status.update(label="✅ All 4 Multi-Agent Tasks Completed Successfully! Synthesizing Answer...", state="complete", expanded=False)
    return response, synthesis_agent

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle pending chip query
pending = st.session_state.pop("pending_chip_query", None)
if pending:
    with st.chat_message("user"):
        st.markdown(pending)
    with st.status("⚡ Multi-Agent Intelligence Network Executing...", expanded=True) as status:
        try:
            response, synthesis_agent = run_orchestrator_with_live_highlights(pending, status)
        except Exception as e:
            status.update(label="❌ Execution Error Occurred", state="error", expanded=True)
            st.error(f"Runtime Error: {e}")
            st.info("Check your `.env` API keys and agent orchestrator imports.")
            st.stop()

    st.session_state.messages.append({"role": "user", "content": pending})
    
    with st.chat_message("assistant"):
        stream_gen = synthesis_agent.synthesize_stream(
            user_query=pending,
            diagnostic_info=response.diagnostic_info,
            fertilizer_info=response.fertilizer_info,
            general_info=response.general_info,
            reflection_result=response.reflection_result
        )
        final_synthesis = st.write_stream(stream_gen)
        response.final_synthesis = final_synthesis

        # Auto-Learning (Second Brain Persistence & FAISS Re-indexing)
        if response and hasattr(response, 'diagnostic_info') and response.diagnostic_info:
            if str(getattr(response.diagnostic_info, 'confidence_level', '')).lower() in ['high', '85%', '90%', '95%']:
                import threading, uuid
                from rag.rag_pipeline import auto_learn_text
                learned_content = f"Farmer Query: {pending}\nDiagnosis: {response.diagnostic_info.suspected_disease}\nRecommended Treatment: {', '.join(response.diagnostic_info.treatment_recommended)}\nVerified Advisory: {final_synthesis}"
                threading.Thread(target=auto_learn_text, args=(learned_content, f"learned_{uuid.uuid4().hex[:8]}"), daemon=True).start()

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

    with st.status("⚡ Multi-Agent Intelligence Network Executing...", expanded=True) as status:
        try:
            response, synthesis_agent = run_orchestrator_with_live_highlights(user_query.strip(), status)
        except Exception as e:
            status.update(label="❌ Execution Error Occurred", state="error", expanded=True)
            st.error(f"Runtime Error: {e}")
            st.info("Check your `.env` API keys and agent orchestrator imports.")
            st.stop()
            
    with st.chat_message("assistant"):
        stream_gen = synthesis_agent.synthesize_stream(
            user_query=user_query.strip(),
            diagnostic_info=response.diagnostic_info,
            fertilizer_info=response.fertilizer_info,
            general_info=response.general_info,
            reflection_result=response.reflection_result
        )
        final_synthesis = st.write_stream(stream_gen)
        response.final_synthesis = final_synthesis

        # Auto-Learning (Second Brain Persistence & FAISS Re-indexing)
        if response and hasattr(response, 'diagnostic_info') and response.diagnostic_info:
            if str(getattr(response.diagnostic_info, 'confidence_level', '')).lower() in ['high', '85%', '90%', '95%']:
                import threading, uuid
                from rag.rag_pipeline import auto_learn_text
                learned_content = f"Farmer Query: {user_query.strip()}\nDiagnosis: {response.diagnostic_info.suspected_disease}\nRecommended Treatment: {', '.join(response.diagnostic_info.treatment_recommended)}\nVerified Advisory: {final_synthesis}"
                threading.Thread(target=auto_learn_text, args=(learned_content, f"learned_{uuid.uuid4().hex[:8]}"), daemon=True).start()

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

            tab_labels = [
                "📋 Advisory Report",
                "📚 Official DOA Guidelines",
                "🛡️ Safety & Regulatory Checks"
            ]

            if dev_mode:
                tab_labels.append("⚡ Developer Agent Trace")

            tabs = st.tabs(tab_labels)

            # TAB 1 — Formatted 5-Section Executive Advisory Report & Download TXT
            with tabs[0]:
                st.markdown(response.final_synthesis)
                st.download_button(
                    label="📥 Download Detailed Executive Advisory (.txt)",
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
                        st.success("✅ ALL CHECKS PASSED — Recommendations comply with Pesticide Act No.33 and Fertilizer Ordinance limits")
                    else:
                        st.warning("⚠️ ATTENTION — Some chemicals may be restricted or require additional precautions")

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
        <div class="glass-card glass-card-hoverable">
            <div class="glass-card-title">🌾 Yala Season (May – August)</div>
            <ul style="margin:0; padding-left:1.2rem;">
                <li>Harvest standing crops promptly to avoid rain.</li>
                <li>Practice dry tillage to break hardpan.</li>
                <li>Apply basal fertilizer at land prep.</li>
            </ul>
        </div>
        <div class="glass-card glass-card-hoverable" style="margin-bottom:0;">
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
        <div class="glass-card glass-card-hoverable" style="padding:0.7rem;">
            <div class="glass-card-title">🍂 Paddy Blast (Pyricularia oryzae)</div>
            <div style="font-size:0.85rem; color:#b7e4c7;">Diamond-shaped spots. Spray Tricyclazole.</div>
        </div>
        <div class="glass-card glass-card-hoverable" style="padding:0.7rem;">
            <div class="glass-card-title">🟤 Brown Spot (Bipolaris oryzae)</div>
            <div style="font-size:0.85rem; color:#b7e4c7;">Oval brown leaf lesions. Apply K & Zn.</div>
        </div>
        <div class="glass-card glass-card-hoverable" style="padding:0.7rem;">
            <div class="glass-card-title">🌿 Sheath Blight (Rhizoctonia)</div>
            <div style="font-size:0.85rem; color:#b7e4c7;">Grey-green patches. Apply Hexaconazole.</div>
        </div>
        <div class="glass-card glass-card-hoverable" style="padding:0.7rem; margin-bottom:0;">
            <div class="glass-card-title">🐛 Brown Planthopper (BPH)</div>
            <div style="font-size:0.85rem; color:#b7e4c7;">Circular yellowing ('hopper burn'). Drain fields.</div>
        </div>
        """, unsafe_allow_html=True)
