"""
Multi-Agent Paddy Disease Diagnostic & Fertilizer Recommendation System
Farmer-Centric Web Application (ui/app.py)

UI: Conversational Chat Interface using Streamlit native chat primitives.
  - Chat history in st.session_state.messages
  - st.chat_message for user & assistant bubbles
  - st.chat_input for queries
  - Each assistant bubble contains interactive 3-tab response
  - Viva mode adds a 4th tab for raw A2A JSON traces
  - Farmer utility tools at bottom
  - Sidebar: status, viva toggle, 1920 hotline, clear button, corpus link
"""

import os
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv
import sys

# Apply nest_asyncio to support run_in_executor/gather in Streamlit threads
import nest_asyncio
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

# ── Custom CSS ──
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 16px; color: #1b4332; }

    .hero-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
        color: #fff; padding: 1.8rem 1.5rem 1.4rem; border-radius: 18px;
        text-align: center; border-bottom: 5px solid #e9c46a;
        box-shadow: 0 8px 30px rgba(27,67,50,0.25); margin-bottom: 0.8rem;
    }
    .hero-title { font-size: 2.0rem; font-weight: 800; letter-spacing: -0.5px; }
    .hero-sub { font-size: 1rem; color: #e9c46a; font-weight: 500; margin-top: 0.3rem; }

    .glass-card {
        background: rgba(248,251,248,0.85); backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(183,228,199,0.5); border-radius: 14px;
        padding: 1.2rem; box-shadow: 0 4px 24px rgba(27,67,50,0.06);
        margin-bottom: 0.8rem;
    }
    .glass-card-title { font-weight: 700; font-size: 1.05rem; color: #1b4332; margin-bottom: 0.4rem; }
    .glass-card p, .glass-card li { font-size: 0.95rem; color: #2d6a4f; line-height: 1.5; }

    .badge-pass {
        background: #d4edda; color: #155724; padding: 0.5rem 1rem;
        border-radius: 22px; font-weight: 700; font-size: 0.95rem;
        display: inline-block; border: 1px solid #c3e6cb; margin-bottom: 0.5rem;
    }
    .badge-warn {
        background: #fff3cd; color: #856404; padding: 0.5rem 1rem;
        border-radius: 22px; font-weight: 700; font-size: 0.95rem;
        display: inline-block; border: 1px solid #ffeeba; margin-bottom: 0.5rem;
    }

    .disease-item {
        background: #f8fbf8; border: 1px solid #d8f3dc; border-radius: 10px;
        padding: 0.7rem 0.9rem; margin-bottom: 0.4rem;
    }
    .disease-name { font-weight: 700; color: #1b4332; font-size: 0.95rem; }
    .disease-desc { font-size: 0.88rem; color: #40916c; margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/rice-plant.png", width=70)
    st.title("🌾 PaddyAgri AI")
    st.caption("Sri Lankan Agriculture Smart Portal")

    st.divider()

    # System Status
    st.subheader("⚙️ System Status")
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if gemini_key:
        st.success("🟢 Google Gemini Ready")
    if groq_key:
        st.success("🟢 Groq Llama Ready")
    if openrouter_key:
        st.info("🔵 OpenRouter Standby")
    if not (gemini_key or groq_key or openrouter_key):
        st.error("🔴 API Key Missing")
        st.info("Set `GEMINI_API_KEY` or `GROQ_API_KEY` in your `.env` file.")

    st.divider()

    # Viva Evaluator Toggle
    viva_mode = st.checkbox(
        "👨‍🏫 Viva Evaluator Mode (Show Raw Agent JSON Trace)",
        value=False
    )
    if viva_mode:
        st.warning("Viva Mode ON — raw A2A JSON traces visible in assistant response tabs.")

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
        if "active_query" in st.session_state:
            st.session_state["active_query"] = ""
        st.rerun()

    st.markdown(
        "🔗 [DOA Advisory PDF Corpus](https://drive.google.com/drive/folders/1O6Teo6_gPBZOd27rtzAI84RSTKKU5er9?usp=sharing)"
    )


# ══════════════════════════════════════════════
# SECTION 1 — HERO BANNER
# ══════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌾 PaddyAgri AI - Sri Lankan Agriculture Smart Portal</div>
    <div class="hero-sub">Smart Paddy Disease Diagnosis & Fertilizer Recommendations for Sri Lankan Farmers</div>
</div>
""", unsafe_allow_html=True)

# ── Quick Suggestion Chips (always visible above chat) ──
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
# SECTION 2 — CHAT HISTORY & MESSAGE RENDERING
# ══════════════════════════════════════════════

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to process user query synchronously (with run_async wrapper)
def run_orchestrator(query: str):
    from core.agent_orchestrator import PaddyAgentOrchestrator
    orchestrator = PaddyAgentOrchestrator()
    return orchestrator.process_user_request(query)

# Handle pending chip query (auto-submit without re-prompt)
pending = st.session_state.pop("pending_chip_query", None)
if pending:
    # Immediately append user message and process
    with st.chat_message("user"):
        st.markdown(pending)
    with st.spinner("🔄 Agents are searching agricultural handbooks in parallel and checking safety rules…"):
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

# Render all chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🌾"):
            # Fixed Output Duplication: Response text is rendered ONLY ONCE inside Tab 1.
            # We no longer execute st.markdown(msg["content"]) outside the tabs.
            response = msg.get("response_obj")
            if response is None:
                st.markdown(msg["content"])
                continue

            # ── build 3-Tab View + optional 4th tab ──
            tab_labels = [
                "📋 ගොවි උපදෙස් පත්‍රිකාව",
                "📚 තහවුරු කළ රාජ්‍ය මූලාශ්‍ර",
                "🛡️ ආරක්‍ෂිත සහ රාජ්‍ය ප්‍රමිතීන්"
            ]
            if viva_mode:
                tab_labels.append("👨‍🏫 Viva Agent JSON Trace")

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

            # TAB 2 — DOA PDF Citations & Page Numbers
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
                    st.info(
                        "No direct handbook citations found for this query. "
                        "The advisory was compiled from general DOA guidelines."
                    )

            # TAB 3 — Pesticide Act & Fertilizer Ordinance Verification
            with tabs[2]:
                st.markdown("### 🛡️ Pesticide Act & Fertilizer Ordinance Compliance")
                refl = response.reflection_result if hasattr(response, "reflection_result") else None
                if refl:
                    if refl.all_checks_passed:
                        st.markdown(
                            '<div class="badge-pass">'
                            "✅ ALL CHECKS PASSED — Recommendations comply with "
                            "Pesticide Act No.33 and Fertilizer Ordinance limits"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="badge-warn">'
                            "⚠️ ATTENTION — Some chemicals may be restricted or "
                            "require additional precautions under current regulations"
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
                    st.info(
                        "This is a general query — no chemical safety or "
                        "regulatory checks were required."
                    )

            # TAB 4 — Viva / Dev JSON Trace (only when toggled)
            if viva_mode:
                with tabs[3]:
                    st.markdown("### 👨‍🏫 Viva Evaluator — Raw A2A Message Trace")
                    intent_val = response.intent.value if hasattr(response, "intent") else "N/A"
                    trace_len = len(response.message_trace) if hasattr(response, "message_trace") else 0
                    st.info(
                        f"**Intent:** {intent_val} · "
                        f"**Messages Exchanged:** {trace_len}"
                    )
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
# SECTION 3 — NATIVE CHAT INPUT
# ══════════════════════════════════════════════

user_query = st.chat_input(
    "Ask your paddy farming question in English, Sinhala, or Singlish...",
    key="chat_input_main"
)

if user_query and user_query.strip():
    # Append & show user message
    st.session_state.messages.append({"role": "user", "content": user_query.strip()})
    with st.chat_message("user"):
        st.markdown(user_query.strip())

    # Run orchestrator
    with st.spinner("🔄 Agents are searching agricultural handbooks in parallel and checking safety rules…"):
        try:
            response = run_orchestrator(user_query.strip())
        except Exception as e:
            st.error(f"Runtime Error: {e}")
            st.info("Check your `.env` API keys and agent orchestrator imports.")
            st.stop()

    # Append assistant response with full object
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.final_synthesis,
        "response_obj": response,
        "query": user_query.strip()
    })
    st.rerun()


# ══════════════════════════════════════════════
# SECTION 4 — PRACTICAL FARMER TOOLS (Bottom)
# ══════════════════════════════════════════════

st.divider()
st.markdown("## 🧰 Farmer Toolkit")

tool_col1, tool_col2, tool_col3 = st.columns(3)

# ── Column 1: Fertilizer Budget Calculator ──
with tool_col1:
    st.markdown("#### 💰 Fertilizer Budget Calculator (LKR)")
    with st.expander("Estimate bags & cost for your field", expanded=True):
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
        st.success(f"**Estimated Cost (subsidized):** LKR {cost_lkr:,.2f}")

# ── Column 2: Seasonal Advisory Card ──
with tool_col2:
    st.markdown("#### 🗓️ Seasonal Advisory")
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">🌾 Yala Season (May – August)</div>
        <ul>
            <li>Harvest standing crops promptly to avoid rain damage.</li>
            <li>Practice dry tillage to break hardpan and control weeds.</li>
            <li>Apply basal fertilizer at land preparation.</li>
        </ul>
    </div>
    <div class="glass-card">
        <div class="glass-card-title">🌧️ Maha Season (September – March)</div>
        <ul>
            <li>Incorporate organic matter during wet ploughing.</li>
            <li>Select DOA-certified high-yield seed varieties.</li>
            <li>Monitor water levels — avoid prolonged flooding.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ── Column 3: Visual Disease Photo Gallery ──
with tool_col3:
    st.markdown("#### 🖼️ Common Paddy Disease Guide")
    with st.expander("Identify common diseases by visual symptoms", expanded=True):
        st.markdown("""
        <div class="disease-item">
            <div class="disease-name">🍂 Paddy Blast (Pyricularia oryzae)</div>
            <div class="disease-desc">
                Diamond-shaped grey-brown spots on leaves with dark borders.
                Severe attacks collapse the leaf mid-rib. Treat with Tricyclazole spray.
            </div>
        </div>
        <div class="disease-item">
            <div class="disease-name">🟤 Brown Spot (Bipolaris oryzae)</div>
            <div class="disease-desc">
                Small oval brown lesions scattered across older leaves.
                Linked to poor soil nutrition. Apply potassium and zinc supplements.
            </div>
        </div>
        <div class="disease-item">
            <div class="disease-name">🌿 Sheath Blight (Rhizoctonia solani)</div>
            <div class="disease-desc">
                Irregular grey-green water-soaked patches on the sheath near waterline.
                Reduce nitrogen and apply Hexaconazole fungicide.
            </div>
        </div>
        <div class="disease-item">
            <div class="disease-name">🐛 Brown Planthopper — BPH</div>
            <div class="disease-desc">
                Plants turn yellow-brown and dry out in circular patches ("hopper burn").
                Drain fields for 3–4 days and use neem-based sprays before chemical control.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──
st.divider()
st.caption(
    "PaddyAgri AI - Sri Lankan Agriculture Smart Portal | "
    "Built with LangChain, FAISS, Gemini, Groq & Streamlit"
)
