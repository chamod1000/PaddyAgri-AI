"""
Multi-Agent Paddy Disease Diagnostic & Fertilizer Recommendation System
Farmer-Centric Web Application (ui/app.py)

Module: IT41043 - Agentic AI (Horizon Campus)
Author: Chamod

Layout (Human-Centered, Above-the-Fold Priority):
  1. Hero Header + Chat Input + Quick Chips    (top — immediate interaction)
  2. Progress Stepper                          (after submit)
  3. 3-Tab Response Area                       (middle — results)
  4. Farmer Utility Tools                      (bottom — calculators, cards, gallery)
  Sidebar: System status, Viva toggle, 1920 hotline, corpus link
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lankan Paddy Farming AI Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS — Paddy Emerald / Mint / Cream
# High contrast, min 16px, glassmorphism cards
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base Typography ── */
    html, body, [class*="css"] {
        font-size: 16px;
        color: #1b4332;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
        color: #ffffff;
        padding: 2.2rem 2rem 1.6rem;
        border-radius: 18px;
        text-align: center;
        border-bottom: 5px solid #e9c46a;
        box-shadow: 0 8px 30px rgba(27, 67, 50, 0.25);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #e9c46a;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    /* ── Glassmorphism Card ── */
    .glass-card {
        background: rgba(248, 251, 248, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(183, 228, 199, 0.5);
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 24px rgba(27, 67, 50, 0.06);
        margin-bottom: 0.8rem;
    }
    .glass-card-title {
        font-weight: 700;
        font-size: 1.08rem;
        color: #1b4332;
        margin-bottom: 0.5rem;
    }
    .glass-card p, .glass-card li {
        font-size: 0.95rem;
        color: #2d6a4f;
        line-height: 1.55;
    }

    /* ── Progress Stepper ── */
    .stepper-row {
        display: flex;
        align-items: stretch;
        gap: 0.6rem;
        margin: 1rem 0 1.5rem;
    }
    .step-card {
        flex: 1;
        background: #f8fbf8;
        border: 2px solid #b7e4c7;
        border-radius: 12px;
        padding: 1rem 0.7rem;
        text-align: center;
        position: relative;
    }
    .step-card.done {
        border-color: #2d6a4f;
        background: #d8f3dc;
    }
    .step-num {
        display: inline-block;
        background: #2d6a4f;
        color: #fff;
        width: 28px; height: 28px;
        line-height: 28px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }
    .step-label {
        font-weight: 700;
        font-size: 0.92rem;
        color: #1b4332;
    }
    .step-detail {
        font-size: 0.78rem;
        color: #555;
        margin-top: 0.15rem;
    }
    .step-arrow {
        display: flex;
        align-items: center;
        font-size: 1.3rem;
        color: #2d6a4f;
        padding: 0 0.1rem;
    }

    /* ── Safety Badges ── */
    .badge-pass {
        background: #d4edda;
        color: #155724;
        padding: 0.55rem 1.1rem;
        border-radius: 22px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #c3e6cb;
        margin-bottom: 0.6rem;
    }
    .badge-warn {
        background: #fff3cd;
        color: #856404;
        padding: 0.55rem 1.1rem;
        border-radius: 22px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #ffeeba;
        margin-bottom: 0.6rem;
    }

    /* ── Disease Gallery Card ── */
    .disease-item {
        background: #f8fbf8;
        border: 1px solid #d8f3dc;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .disease-name {
        font-weight: 700;
        color: #1b4332;
        font-size: 1rem;
    }
    .disease-desc {
        font-size: 0.9rem;
        color: #40916c;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/rice-plant.png", width=70)
    st.title("🌾 Paddy Farming AI")
    st.caption("Horizon Campus — IT41043 Assignment")

    st.divider()

    # System Status
    st.subheader("⚙️ System Status")
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if groq_key or openrouter_key:
        st.success("🟢 AI Services Online")
    else:
        st.error("🔴 API Key Missing")
        st.info("Set `GROQ_API_KEY` or `OPENROUTER_API_KEY` in your `.env` file.")

    st.divider()

    # Viva Evaluator Toggle
    viva_mode = st.checkbox(
        "👨‍🏫 Viva Evaluator Mode (Show Raw Agent JSON Trace)",
        value=False
    )
    if viva_mode:
        st.warning("Viva Mode ON — raw A2A JSON traces visible in Tab 4.")

    st.divider()

    # 📞 1920 Agrarian Hotline
    st.markdown("### 📞 Agrarian Helpline")
    st.info(
        "**Department of Agriculture — 1920**\n\n"
        "☎️ **1920** (Govi Sahana Piyasa)\n\n"
        "🌐 [doa.gov.lk](http://www.doa.gov.lk)\n\n"
        "*Free expert advice for Sri Lankan farmers.*"
    )

    st.divider()
    st.markdown(
        "🔗 [DOA Advisory PDF Corpus (Google Drive)]"
        "(https://drive.google.com/drive/folders/1O6Teo6_gPBZOd27rtzAI84RSTKKU5er9?usp=sharing)"
    )


# ══════════════════════════════════════════════
# SECTION 1 — HERO HEADER + CHAT INPUT (Above the Fold)
# ══════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌾 Sri Lankan Paddy Farming AI Advisor</div>
    <div class="hero-sub">Smart Paddy Disease Diagnosis & Fertilizer Recommendations for Sri Lankan Farmers</div>
</div>
""", unsafe_allow_html=True)

# Session state
if "active_query" not in st.session_state:
    st.session_state["active_query"] = ""

# Chat input — immediately visible
user_input = st.text_area(
    "Ask your farming question in Simple English:",
    value=st.session_state["active_query"],
    placeholder="Example: My paddy leaves have brown spots and are drying up. What should I do?",
    height=100,
    key="main_query_input"
)

col_btn1, col_btn2 = st.columns([4, 1])
submit_button = col_btn1.button("🚀 Get AI Advisory", type="primary", use_container_width=True)
clear_button = col_btn2.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.session_state["active_query"] = ""
    st.rerun()

# Quick Suggestion Chips
st.markdown("##### 💡 Quick Suggestions — click any chip to auto-fill your question:")
chip1, chip2, chip3, chip4 = st.columns(4)

if chip1.button("🍂 Paddy Blast Symptoms & Treatments", use_container_width=True):
    st.session_state["active_query"] = (
        "What are the common symptoms of Paddy Blast disease and what chemical "
        "and organic treatments does the DOA recommend?"
    )
    st.rerun()

if chip2.button("🌱 Yala Season NPK Fertilizer Rates", use_container_width=True):
    st.session_state["active_query"] = (
        "What are the recommended Urea, TSP, and MOP fertilizer rates per acre "
        "for the Yala crop season in Sri Lanka?"
    )
    st.rerun()

if chip3.button("🐛 Brown Planthopper (BPH) Pest Control", use_container_width=True):
    st.session_state["active_query"] = (
        "How do I identify and control Brown Planthopper (BPH) pest attacks "
        "in paddy fields using DOA certified methods?"
    )
    st.rerun()

if chip4.button("📜 Certified Seed Paddy Standards", use_container_width=True):
    st.session_state["active_query"] = (
        "What quality standards, purity percentages, and germination rates are "
        "required for Certified Seed Paddy in Sri Lanka?"
    )
    st.rerun()


# ══════════════════════════════════════════════
# SECTION 2 & 3 — PROCESS STEPPER + RESPONSE TABS
# ══════════════════════════════════════════════

query_to_run = user_input.strip() if user_input.strip() else st.session_state.get("active_query", "")

if (submit_button or st.session_state["active_query"]) and query_to_run:
    with st.spinner("🔄 Agents are searching agricultural handbooks and checking safety rules…"):
        try:
            from core.agent_orchestrator import PaddyAgentOrchestrator
            orchestrator = PaddyAgentOrchestrator()
            response = orchestrator.process_user_request(query_to_run)

            st.success(f'✅ AI Advisory ready for: "{query_to_run}"')

            # ── Progress Stepper ──
            st.markdown("""
            <div class="stepper-row">
                <div class="step-card done">
                    <div class="step-num">1</div><br>
                    <div class="step-label">🔍 Identifying Query</div>
                    <div class="step-detail">RouterAgent ✔</div>
                </div>
                <div class="step-arrow">➜</div>
                <div class="step-card done">
                    <div class="step-num">2</div><br>
                    <div class="step-label">📚 Searching DOA PDF Corpus</div>
                    <div class="step-detail">RAG Vector Search ✔</div>
                </div>
                <div class="step-arrow">➜</div>
                <div class="step-card done">
                    <div class="step-num">3</div><br>
                    <div class="step-label">🛡️ Safety Verification</div>
                    <div class="step-detail">ReflectionAgent ✔</div>
                </div>
                <div class="step-arrow">➜</div>
                <div class="step-card done">
                    <div class="step-num">4</div><br>
                    <div class="step-label">📄 Generating Advisory</div>
                    <div class="step-detail">Synthesis ✔</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Tabbed Response Area ──
            tab_labels = [
                "📋 Farmer Advisory Report",
                "📚 Official DOA Guidelines",
                "🛡️ Safety & Regulatory Checks",
            ]
            if viva_mode:
                tab_labels.append("👨‍🏫 Viva Agent JSON Trace")

            tabs = st.tabs(tab_labels)

            # TAB 1 — Advisory Report
            with tabs[0]:
                st.markdown("### 📋 Farmer Advisory & Recommended Action Plan")
                st.markdown(response.final_synthesis)
                st.download_button(
                    label="📥 Download Advisory Report (.txt)",
                    data=response.final_synthesis,
                    file_name="paddy_advisory_report.txt",
                    mime="text/plain"
                )

            # TAB 2 — DOA Guidelines & RAG Citations
            with tabs[1]:
                st.markdown("### 📚 Official DOA Handbook Citations")
                all_sources = []
                if response.diagnostic_info:
                    all_sources.extend(response.diagnostic_info.rag_sources)
                if response.fertilizer_info:
                    all_sources.extend(response.fertilizer_info.rag_sources)

                if all_sources:
                    for i, src in enumerate(all_sources, 1):
                        with st.expander(
                            f"📄 Source #{i} — {src.filename} · Page {src.page}"
                        ):
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

            # TAB 3 — Safety & Regulatory Checks
            with tabs[2]:
                st.markdown("### 🛡️ Pesticide Act & Fertilizer Ordinance Compliance")
                refl = response.reflection_result
                if refl:
                    if refl.all_checks_passed:
                        st.markdown(
                            '<div class="badge-pass">'
                            '✅ ALL CHECKS PASSED — Recommendations comply with '
                            'Pesticide Act No.33 and Fertilizer Ordinance limits'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="badge-warn">'
                            '⚠️ ATTENTION — Some chemicals may be restricted or '
                            'require additional precautions under current regulations'
                            '</div>',
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
                    st.info(
                        f"**Intent:** {response.intent.value} · "
                        f"**Messages Exchanged:** {len(response.message_trace)}"
                    )
                    for idx, msg in enumerate(response.message_trace, 1):
                        with st.expander(
                            f"✉️ #{idx}: {msg.sender} ➔ {msg.receiver} "
                            f"[{msg.intent.value}]",
                            expanded=True
                        ):
                            st.json({
                                "message_id": msg.message_id,
                                "sender": msg.sender,
                                "receiver": msg.receiver,
                                "intent": msg.intent.value,
                                "timestamp": msg.timestamp,
                                "payload": msg.payload,
                            })

        except Exception as e:
            st.error(f"Runtime Error: {e}")
            st.info(
                "Check that your `.env` file contains a valid API key and "
                "the agent orchestrator modules are importable."
            )


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


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Horizon Campus — Module IT41043: Agentic AI Assignment · "
    "Built with LangChain, FAISS, Groq & Streamlit"
)
