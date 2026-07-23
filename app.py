"""
Multi-Agent Paddy Disease Diagnostic & Fertilizer Recommendation System
Modern Sri Lankan Farmer Web Application (app.py)

Module: IT41043 - Agentic AI (Horizon Campus)
Author: Chamod
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="PaddyAgri AI - Sri Lankan Multi-Agent System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Emerald & Paddy Gold Aesthetic
st.markdown("""
<style>
    /* Global Container Styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0b3c26;
        background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 50%, #95d5b2 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 16px;
        border-left: 8px solid #2d6a4f;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #1b4332;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    
    /* Feature Highlight Cards */
    .feature-box {
        background-color: #f4fbf7;
        border: 1px solid #b7e4c7;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .feature-title {
        font-weight: 700;
        color: #2d6a4f;
        font-size: 1.05rem;
    }
    
    /* Safety Badge Styling */
    .badge-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-warn {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/rice-plant.png", width=75)
    st.title("🌾 PaddyAgri AI")
    st.caption("Horizon Campus - IT41043 Assignment")
    
    st.divider()
    st.subheader("⚙️ System Connectivity")
    
    # API Status Check
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if groq_key or openrouter_key:
        st.success("🟢 API Provider Connected (Groq / OpenRouter)")
    else:
        st.error("🔴 API Key Missing")
        st.info("Please set `GROQ_API_KEY` in your `.env` file.")

    st.markdown("""
    **Model Selection Architecture:**
    - ⚡ **Router Model:** `Groq (Llama 3.1 8B)`
    - 🧠 **Reasoning Model:** `Groq (Llama 3.3 70B)`
    - 📚 **RAG Engine:** `FAISS + Multilingual MiniLM`
    """)

    st.divider()
    
    # 📞 Agrarian Hotline Card
    st.markdown("### 📞 Agrarian Helpline")
    st.info("""
    **Department of Agriculture Helpline:**
    ☎️ **1920** (Govi Sahana Piyasa)  
    🌐 [Agri Dept Sri Lanka](http://www.doa.gov.lk)
    """)

    st.divider()
    st.markdown("🔗 [Full PDF Corpus on Google Drive](https://drive.google.com/drive/folders/1O6Teo6_gPBZOd27rtzAI84RSTKKU5er9?usp=sharing)")


# Header Banner
st.markdown("""
<div class="main-header">
    <div>🌾 ශ්‍රී ලාංකික ගොවිජන මල්ටි-ඒජන්ට් AI පද්ධතිය</div>
    <div class="sub-header">Sri Lankan Paddy Disease Diagnostic & Fertilizer Recommendation System</div>
</div>
""", unsafe_allow_html=True)


# Top Interactive Practical Widgets
col_widget1, col_widget2, col_widget3 = st.columns([1.2, 1, 1])

with col_widget1:
    st.markdown("#### 💰 Fertilizer Cost Calculator (LKR)")
    with st.expander("Calculate Fertilizer Budget for your Field", expanded=False):
        acres = st.number_input("Field Size (Acres):", min_value=0.5, max_value=50.0, value=2.0, step=0.5)
        calc_season = st.selectbox("Season:", ["Yala Season (යල)", "Maha Season (මහ)"])
        
        if "Yala" in calc_season:
            urea_bags = round((acres * 50) / 50, 1)
            tsp_bags = round((acres * 25) / 50, 1)
            mop_bags = round((acres * 25) / 50, 1)
        else:
            urea_bags = round((acres * 65) / 50, 1)
            tsp_bags = round((acres * 30) / 50, 1)
            mop_bags = round((acres * 30) / 50, 1)

        cost_lkr = (urea_bags + tsp_bags + mop_bags) * 2500
        
        st.write(f"• **Urea (50kg bags):** {urea_bags} bags")
        st.write(f"• **TSP (50kg bags):** {tsp_bags} bags")
        st.write(f"• **MOP (50kg bags):** {mop_bags} bags")
        st.success(f"**Est. Subsidized Cost:** LKR {cost_lkr:,.2f}")

with col_widget2:
    st.markdown("#### 🗓️ Season Advisor")
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">🌾 Current Season: Late Yala</div>
        <p style="margin-top:0.4rem; font-size:0.88rem;">Harvesting & Land Preparation for Maha Season. Ensure field drainage & seed paddy purity testing.</p>
    </div>
    """, unsafe_allow_html=True)

with col_widget3:
    st.markdown("#### 🖼️ Visual Disease Matcher")
    with st.expander("Paddy Disease Gallery", expanded=False):
        st.markdown("- **Paddy Blast (ගොයම් පාළු):** Spindle-shaped lesions on leaves with brown borders.")
        st.markdown("- **Brown Planthopper (පැළ මැක්සා):** Hoppercurn patches of drying yellow plants.")
        st.markdown("- **Sheath Blight:** Oval water-soaked spots on leaf sheaths near water line.")


st.divider()

# Session State for Active Query
if "active_query" not in st.session_state:
    st.session_state["active_query"] = ""

# Sample Quick Queries
st.markdown("##### 💡 Sample Quick Queries (Click to run):")
col_q1, col_q2, col_q3, col_q4 = st.columns(4)

if col_q1.button("🍂 Paddy Blast Symptoms"):
    st.session_state["active_query"] = "What are the common symptoms and control measures for Paddy Blast disease?"

if col_q2.button("🌱 Yala Season Fertilizer"):
    st.session_state["active_query"] = "What is the recommended NPK fertilizer mixture for Yala season paddy in Polonnaruwa?"

if col_q3.button("🇱🇰 Sinhala Query (දුඹුරු ලප)"):
    st.session_state["active_query"] = "ගොයම් පත්‍ර වල දුඹුරු පැහැ ලප ඇති වී ඇත, යල කන්නයට පොහොර යොදන්නේ කෙසේද?"

if col_q4.button("📜 Seed Paddy Standards"):
    st.session_state["active_query"] = "What are the germination requirements for certified seed paddy in Sri Lanka?"


# Input Text Box
user_input = st.text_area(
    "Enter your agricultural query (in English or Sinhala / ඉංග්‍රීසි හෝ සිංහලෙන් අසන්න):",
    value=st.session_state["active_query"],
    placeholder="e.g. My paddy field leaves are turning yellow with brown spots. What fertilizer and treatment should I use?",
    height=100
)

col_btn1, col_btn2 = st.columns([3, 1])
submit_button = col_btn1.button("🚀 Run Multi-Agent Diagnostic Analysis", type="primary", use_container_width=True)
clear_button = col_btn2.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.session_state["active_query"] = ""
    st.rerun()

# Execution & Response Render
query_to_run = user_input.strip() if user_input.strip() else st.session_state["active_query"]

if (submit_button or st.session_state["active_query"]) and query_to_run:
    with st.spinner("🔄 Multi-Agent System orchestrating diagnosis, RAG search & reflection verification..."):
        try:
            from agent_orchestrator import PaddyAgentOrchestrator
            orchestrator = PaddyAgentOrchestrator()
            response = orchestrator.process_user_request(query_to_run)
            
            st.success(f"✅ Multi-Agent Analysis Completed for: \"{query_to_run}\"")
            
            tab_report, tab_agents, tab_rag, tab_safety = st.tabs([
                "📋 Farmer Advisory Report", 
                "🤖 Live Agent Message Trace", 
                "📚 RAG Document Citations", 
                "🛡️ Safety & Regulatory Verification"
            ])
            
            # TAB 1: Advisory Report
            with tab_report:
                st.markdown(response.final_synthesis)
                st.download_button(
                    label="📥 Download Farmer Advisory Report (TXT)",
                    data=response.final_synthesis,
                    file_name="paddy_advisory_report.txt",
                    mime="text/plain"
                )
            
            # TAB 2: Agent Communication Trace
            with tab_agents:
                st.markdown("### 🔄 Real-time Agent Communication Flow")
                st.info(f"Classified Intent: **{response.intent.value}** | Total Agent Messages Exchanged: **{len(response.message_trace)}**")
                
                for idx, msg in enumerate(response.message_trace, 1):
                    with st.expander(f"✉️ Message #{idx}: {msg.sender} ➔ {msg.receiver} [{msg.intent.value}]", expanded=True):
                        st.json({
                            "message_id": msg.message_id,
                            "sender": msg.sender,
                            "receiver": msg.receiver,
                            "intent": msg.intent.value,
                            "timestamp": msg.timestamp,
                            "payload": msg.payload
                        })

            # TAB 3: RAG Document Citations
            with tab_rag:
                st.markdown("### 📚 Retrieved RAG Knowledge Base Chunks")
                
                all_sources = []
                if response.diagnostic_info:
                    all_sources.extend(response.diagnostic_info.rag_sources)
                if response.fertilizer_info:
                    all_sources.extend(response.fertilizer_info.rag_sources)
                
                if all_sources:
                    for i, src in enumerate(all_sources, 1):
                        with st.expander(f"📄 Citation #{i}: [{src.category}] {src.filename} (Page {src.page})"):
                            st.write(f"**Category:** `{src.category}` | **Filename:** `{src.filename}` | **Page:** {src.page}")
                            st.markdown(f"> *\"{src.content}\"*")
                else:
                    st.write("No direct RAG vector search chunks retrieved for this query.")

            # TAB 4: Safety Verification
            with tab_safety:
                st.markdown("### 🛡️ Reflection & Self-Critique Verification Results")
                refl = response.reflection_result
                if refl:
                    if refl.all_checks_passed:
                        st.markdown('<span class="badge-pass">✅ ALL SAFETY & REGULATORY CHECKS PASSED</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-warn">⚠️ SAFETY WARNINGS DETECTED</span>', unsafe_allow_html=True)
                    
                    st.markdown("#### Safety Check Verdicts:")
                    for v in refl.verdicts:
                        icon = "✅" if v.passed else "⚠️"
                        st.write(f"{icon} **{v.check_name}**: {v.message}")
                    
                    st.markdown("#### Department of Agriculture Regulatory Citations:")
                    for cite in refl.regulatory_citations:
                        st.write(f"- 📜 {cite}")
                else:
                    st.write("No safety reflection required for general queries.")
                    
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.info("Make sure `GROQ_API_KEY` is configured in your local `.env` file.")


# Footer
st.divider()
st.caption("Horizon Campus — Module IT41043: Agentic AI Assignment | Built with LangChain, FAISS, Groq & Streamlit")
