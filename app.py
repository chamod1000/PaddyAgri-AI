"""
Multi-Agent Paddy Disease Diagnostic & Fertilizer Recommendation System
Streamlit Web Application (app.py)

Module: IT41043 - Agentic AI (Horizon Campus)
Author: Chamod
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="PaddyAgri AI - Sri Lankan Multi-Agent System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Visual Excellence
st.markdown("""
<style>
    /* Theme Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b4332;
        background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        border-left: 6px solid #2d6a4f;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #40916c;
        font-weight: 500;
        margin-top: -0.5rem;
    }
    
    /* Card design */
    .stat-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    
    /* Agent Message Trace styling */
    .agent-msg-box {
        background-color: #f1f8f5;
        border-left: 4px solid #52b788;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    
    /* Safety Badge Styling */
    .badge-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warn {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/rice-plant.png", width=70)
    st.title("🌾 PaddyAgri AI")
    st.caption("Horizon Campus - IT41043 Assignment")
    
    st.divider()
    st.subheader("⚙️ System Status")
    
    # Check API Keys
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if groq_key or openrouter_key:
        st.success("🟢 API Provider Connected")
    else:
        st.error("🔴 API Key Missing (.env)")
        st.warning("Please configure GROQ_API_KEY or OPENROUTER_API_KEY in .env")

    st.markdown("""
    **Model Selection Architecture:**
    - ⚡ **Router Model:** `Groq (Llama 3.1 8B Instant)`
    - 🧠 **Reasoning Model:** `OpenRouter (Claude 3.5) / Groq (Llama 70B)`
    - 📚 **RAG Engine:** `FAISS + Multilingual MiniLM`
    """)

    st.divider()
    st.markdown("### 📂 Corpus Categories")
    st.markdown("""
    - 🟢 Disease & Pests (20+ PDFs)
    - 🟢 Fertilizer & Soil Nutrients
    - 🟢 Seed Paddy Standards
    - 🟢 Quarantine & Import Regulations
    """)
    st.markdown("🔗 [Full Dataset on Google Drive](https://drive.google.com/drive/folders/1O6Teo6_gPBZOd27rtzAI84RSTKKU5er9?usp=sharing)")


# Main Page Header
st.markdown("""
<div class="main-header">
    <div>🌾 Sri Lankan Paddy Disease Diagnostic & Fertilizer Recommendation System</div>
    <div class="sub-header">Multi-Agent AI Advisory System for Farmers & Agricultural Extension Officers</div>
</div>
""", unsafe_allow_html=True)


# Sample Quick Queries
st.markdown("##### 💡 Sample Quick Queries (Click to try):")
col_q1, col_q2, col_q3, col_q4 = st.columns(4)

sample_query = ""

if col_q1.button("🍂 Paddy Blast Symptoms"):
    sample_query = "What are the common symptoms and control measures for Paddy Blast disease?"

if col_q2.button("🌱 Yala Season Fertilizer"):
    sample_query = "What is the recommended NPK fertilizer mixture for Yala season paddy in Polonnaruwa?"

if col_q3.button("🇱🇰 Sinhala Query (දුඹුරු ලප)"):
    sample_query = "ගොයම් පත්‍ර වල දුඹුරු පැහැ ලප ඇති වී ඇත, යල කන්නයට පොහොර යොදන්නේ කෙසේද?"

if col_q4.button("📜 Seed Paddy Standards"):
    sample_query = "What are the germination requirements for certified seed paddy in Sri Lanka?"


# Input Form
with st.form("query_form", clear_on_submit=False):
    user_input = st.text_area(
        "Enter your agricultural query (in English or Sinhala / ඉංග්‍රීසි හෝ සිංහලෙන් අසන්න):",
        value=sample_query if sample_query else "",
        placeholder="e.g. My paddy field leaves are turning yellow with brown spots. What fertilizer and treatment should I use?",
        height=100
    )
    submit_button = st.form_submit_button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)


# Execution & Response Render
if submit_button and user_input.strip():
    with st.spinner("🔄 Multi-Agent System orchestrating diagnosis & retrieval..."):
        try:
            from agent_orchestrator import PaddyAgentOrchestrator
            orchestrator = PaddyAgentOrchestrator()
            response = orchestrator.process_user_request(user_input.strip())
            
            # Display Final Synthesis
            st.success("✅ Multi-Agent Analysis Complete!")
            
            tab_report, tab_agents, tab_rag, tab_safety = st.tabs([
                "📋 Farmer Advisory Report", 
                "🤖 Agent-to-Agent Message Trace", 
                "📚 RAG Document Citations", 
                "🛡️ Safety & Regulatory Verification"
            ])
            
            # TAB 1: Report
            with tab_report:
                st.markdown(response.final_synthesis)
                
                # Downloadable Advisory Report Button
                st.download_button(
                    label="📥 Download Advisory Report (TXT)",
                    data=response.final_synthesis,
                    file_name="paddy_advisory_report.txt",
                    mime="text/plain"
                )
            
            # TAB 2: Agent Communication Trace
            with tab_agents:
                st.markdown("### 🔄 Real-time Agent Message Flow")
                st.info(f"Classified Intent: **{response.intent.value}** | Total Messages Exchanged: **{len(response.message_trace)}**")
                
                for idx, msg in enumerate(response.message_trace, 1):
                    with st.expander(f"✉️ Message #{idx}: {msg.sender} ➔ {msg.receiver} ({msg.intent.value})", expanded=True):
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
                st.markdown("### 📚 Retrieved Domain Knowledge Chunks")
                
                all_sources = []
                if response.diagnostic_info:
                    all_sources.extend(response.diagnostic_info.rag_sources)
                if response.fertilizer_info:
                    all_sources.extend(response.fertilizer_info.rag_sources)
                
                if all_sources:
                    for i, src in enumerate(all_sources, 1):
                        with st.expander(f"📄 Citation #{i}: [{src.category}] {src.filename} (Page {src.page}) - Distance: {src.score:.4f}"):
                            st.write(f"**Category:** `{src.category}` | **Filename:** `{src.filename}` | **Page:** {src.page}")
                            st.markdown(f"> *\"{src.content}\"*")
                else:
                    st.write("No direct RAG vector search chunks retrieved for this query.")

            # TAB 4: Safety & Regulatory Verification
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
            st.info("Make sure you have created your `.env` file with valid `GROQ_API_KEY` or `OPENROUTER_API_KEY`.")

# Footer
st.divider()
st.caption("Horizon Campus — Module IT41043: Agentic AI Assignment | Built with LangChain, FAISS, Groq, OpenRouter & Streamlit")
