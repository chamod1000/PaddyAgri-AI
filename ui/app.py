"""
Multi-Agent Paddy Disease Diagnostic & Fertilizer Recommendation System
Farmer-Centric Web Application (ui/app.py)

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
    page_title="PaddyAgri AI - Sri Lankan Paddy Advisory System",
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
    
    /* Progress Stepper Styling */
    .step-box {
        background-color: #ffffff;
        border: 1px solid #c7f9cc;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(45, 106, 79, 0.05);
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
    st.caption(" Horizon Campus — IT41043 Assignment")
    
    st.divider()
    st.subheader("⚙️ පද්ධති සක්‍රීයතාව (System Status)")
    
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if groq_key or openrouter_key:
        st.success("🟢 AI සේවාව සක්‍රීයයි (AI System Ready)")
    else:
        st.error("🔴 API Key එක නොමැත")
        st.info("කරුණාකර `.env` ගොනුවේ `GROQ_API_KEY` සටහන් කරන්න.")

    st.divider()
    
    # 👨‍🏫 Viva Evaluator / Developer Mode Toggle
    viva_mode = st.checkbox("👨‍🏫 Viva / Dev Mode (Show Raw Agent JSON Trace)", value=False)
    if viva_mode:
        st.warning("Viva Evaluator Mode Enabled: Technical JSON Message Traces will be displayed.")

    st.divider()
    
    # 📞 Agrarian Hotline Card
    st.markdown("### 📞 ගොවිජන සහන සේවය")
    st.info("""
    **කෘෂිකර්ම දෙපාර්තමේන්තු ක්ෂණික ඇමතුම්:**
    ☎️ **1920** (ගොවි සහන පියස)  
    🌐 [කෘෂිකර්ම දෙපාර්තමේන්තුව](http://www.doa.gov.lk)
    """)

    st.divider()
    st.markdown("🔗 [කෘෂිකාර්මික උපදෙස් PDF මූලාශ්‍ර (Google Drive)](https://drive.google.com/drive/folders/1O6Teo6_gPBZOd27rtzAI84RSTKKU5er9?usp=sharing)")


# Header Banner
st.markdown("""
<div class="main-header">
    <div>🌾 ශ්‍රී ලාංකික ගොවිජන AI උපදේශක පද්ධතිය</div>
    <div class="sub-header">Sri Lankan Paddy Disease Diagnostic & Fertilizer Advisory Portal</div>
</div>
""", unsafe_allow_html=True)


# Top Interactive Practical Widgets
col_widget1, col_widget2, col_widget3 = st.columns([1.2, 1, 1])

with col_widget1:
    st.markdown("#### 💰 පොහොර වියදම් ගණකය (LKR Calculator)")
    with st.expander("ඔබේ කුඹුරට අවශ්‍ය පොහොර ප්‍රමාණය හා මුදල ගණනය කරන්න", expanded=False):
        acres = st.number_input("කුඹුරේ ප්‍රමාණය (අක්කර):", min_value=0.5, max_value=50.0, value=2.0, step=0.5)
        calc_season = st.selectbox("වගා කන්නය:", ["යල කන්නය (Yala Season)", "මහ කන්නය (Maha Season)"])
        
        if "යල" in calc_season:
            urea_bags = round((acres * 50) / 50, 1)
            tsp_bags = round((acres * 25) / 50, 1)
            mop_bags = round((acres * 25) / 50, 1)
        else:
            urea_bags = round((acres * 65) / 50, 1)
            tsp_bags = round((acres * 30) / 50, 1)
            mop_bags = round((acres * 30) / 50, 1)

        cost_lkr = (urea_bags + tsp_bags + mop_bags) * 2500
        
        st.write(f"• **යූරියා (50kg මලු):** {urea_bags} යි")
        st.write(f"• **TSP (50kg මලු):** {tsp_bags} යි")
        st.write(f"• **MOP (50kg මලු):** {mop_bags} යි")
        st.success(f"**අනුමාන සහනාධාර පිරිවැය:** LKR {cost_lkr:,.2f}")

with col_widget2:
    st.markdown("#### 🗓️ කන්නයේ වගා උපදේශක")
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">🌾 වත්මන් කාලසීමාව: යල කන්නයේ අස්වැන්න නෙලීම</div>
        <p style="margin-top:0.4rem; font-size:0.88rem;">මහ කන්නය සඳහා බිම් සකස් කිරීම සහ පිරිසිදු බිත්තර වී තෝරා ගැනීම සිදුකරන්න.</p>
    </div>
    """, unsafe_allow_html=True)

with col_widget3:
    st.markdown("#### 🖼️ ගොයම් රෝග හඳුනාගැනීම")
    with st.expander("ප්‍රධාන ගොයම් රෝග හඳුනාගන්න", expanded=False):
        st.markdown("- **ගොයම් පාළු රෝගය (Blast):** පත්‍ර වල කපොලු හැඩැති දුඹුරු දාර සහිත ලප.")
        st.markdown("- **පැළ මැක්සා (BPH):** ගොයම් පඳුරු කහ පැහැ වී වේලී යාම.")
        st.markdown("- **කොළ පාළු රෝගය (Sheath Blight):** කඳ ආශ්‍රිතව තෙතමනය සහිත පාලු ලප.")


st.divider()

# Session State for Active Query
if "active_query" not in st.session_state:
    st.session_state["active_query"] = ""

# Farmer Interactive Symptom Selector
st.markdown("##### 💡 ක්ෂණික ප්‍රශ්න සහ රෝග ලක්ෂණ තෝරන්න (Select Symptoms):")
col_q1, col_q2, col_q3, col_q4 = st.columns(4)

if col_q1.button("🍂 ගොයම් පාළු රෝග ලක්ෂණ"):
    st.session_state["active_query"] = "ගොයම් පත්‍ර වල දුඹුරු පාළු ලප ඇති වී ඇත. මෙයට ප්‍රතිකාර සහ පාලන ක්‍රම මොනවාද?"

if col_q2.button("🌱 යල කන්නයේ පොහොර මාත්‍රාව"):
    st.session_state["active_query"] = "පොළොන්නරුව දිස්ත්‍රික්කයේ යල කන්නයේ ගොයම් සඳහා නිර්දේශිත යූරියා, TSP සහ MOP පොහොර ප්‍රමාණය කොපමණද?"

if col_q3.button("🐛 පැළ මැක්සා හා පළිබෝධ"):
    st.session_state["active_query"] = "ගොයම් පඳුරු කහ පැහැ වී කෘමීන් බෝ වී ඇත. පාලනය කරන්නේ කෙසේද?"

if col_q4.button("📜 සහතික කළ බිත්තර වී"):
    st.session_state["active_query"] = "ශ්‍රී ලංකාවේ සහතික කළ බිත්තර වී සඳහා පැළවීමේ ප්‍රතිශතය සහ ප්‍රමිතීන් මොනවාද?"


# Input Text Box
user_input = st.text_area(
    "ඔබගේ ගොවිතැන් ගැටලුව මෙහි සටහන් කරන්න (සිංහලෙන් හෝ ඉංග්‍රීසියෙන්):",
    value=st.session_state["active_query"],
    placeholder="උදා: මගේ ගොයම් කොළ කහ වී දුඹුරු ලප ඇති වී ඇත. යෙදිය යුතු පොහොර සහ ඖෂධ මොනවාද?",
    height=100
)

col_btn1, col_btn2 = st.columns([3, 1])
submit_button = col_btn1.button("🚀 AI උපදෙස් ලබාගන්න (Get AI Advisory)", type="primary", use_container_width=True)
clear_button = col_btn2.button("🗑️ මකන්න (Clear)", use_container_width=True)

if clear_button:
    st.session_state["active_query"] = ""
    st.rerun()

# Execution & Response Render
query_to_run = user_input.strip() if user_input.strip() else st.session_state["active_query"]

if (submit_button or st.session_state["active_query"]) and query_to_run:
    with st.spinner("🔄 AI නියෝජිතයින් (Multi-Agents) ඔබේ ගැටලුව පරීක්ෂා කරමින් පවතී..."):
        try:
            from core.agent_orchestrator import PaddyAgentOrchestrator
            orchestrator = PaddyAgentOrchestrator()
            response = orchestrator.process_user_request(query_to_run)
            
            st.success(f"✅ \"{query_to_run}\" සඳහා කෘෂිකාර්මික AI විශ්ලේෂණය සාර්ථකව නිම විය!")
            
            # Farmer Friendly Progress Stepper Overview
            st.markdown("#### 🔄 AI පද්ධතියේ ක්‍රියාකාරී පියවර (Process Summary):")
            st_col1, st_col2, st_col3, st_col4 = st.columns(4)
            st_col1.markdown('<div class="step-box"><b>1. 🔍 ප්‍රශ්නය හඳුනාගැනීම</b><br><small>RouterAgent</small></div>', unsafe_allow_html=True)
            st_col2.markdown('<div class="step-box"><b>2. 📚 පොත්පත් පරීක්ෂාව</b><br><small>RAG Vector Search</small></div>', unsafe_allow_html=True)
            st_col3.markdown('<div class="step-box"><b>3. 🛡️ සුරක්ෂිතතා පරීක්ෂාව</b><br><small>ReflectionAgent</small></div>', unsafe_allow_html=True)
            st_col4.markdown('<div class="step-box"><b>4. 📄 උපදෙස් සකස් කිරීම</b><br><small>Final Synthesis</small></div>', unsafe_allow_html=True)

            st.divider()

            # Main Farmer Friendly Tabs
            tab_names = [
                "📋 ගොවි උපදෙස් පත්‍රිකාව (Advisory Report)", 
                "📚 තහවුරු කළ රාජ්‍ය මූලාශ්‍ර (DOA Guidelines)", 
                "🛡️ ආරක්‍ෂිත සහ රාජ්‍ය ප්‍රමිතීන් (Safety Checks)"
            ]
            if viva_mode:
                tab_names.append("👨‍🏫 Live Agent JSON Trace (Viva Mode)")

            tabs = st.tabs(tab_names)
            
            # TAB 1: Advisory Report
            with tabs[0]:
                st.markdown(response.final_synthesis)
                st.download_button(
                    label="📥 ගොවි උපදෙස් පත්‍රිකාව ඩවුන්ලෝඩ් කරන්න (Download Advisory Text)",
                    data=response.final_synthesis,
                    file_name="paddy_advisory_report.txt",
                    mime="text/plain"
                )
            
            # TAB 2: Official DOA Guidelines & Citations
            with tabs[1]:
                st.markdown("### 📚 කෘෂිකර්ම දෙපාර්තමේන්තුවේ තහවුරු කළ මූලාශ්‍ර")
                
                all_sources = []
                if response.diagnostic_info:
                    all_sources.extend(response.diagnostic_info.rag_sources)
                if response.fertilizer_info:
                    all_sources.extend(response.fertilizer_info.rag_sources)
                
                if all_sources:
                    for i, src in enumerate(all_sources, 1):
                        with st.expander(f"📄 රාජ්‍ය නිල ප්‍රකාශනය #{i}: {src.filename} (පිටුව {src.page})"):
                            st.write(f"**අංශය:** `{src.category}` | **ලේඛනය:** `{src.filename}` | **පිටුව:** {src.page}")
                            st.markdown(f"> *\"{src.content}\"*")
                else:
                    st.write("මෙම ප්‍රශ්නය සඳහා සෘජු PDF ඡේද හමු නොවූ අතර සාමාන්‍ය උපදෙස් අනුගමනය කෙරිණි.")

            # TAB 3: Safety & Regulations
            with tabs[2]:
                st.markdown("### 🛡️ කෘෂිකාර්මික සුරක්ෂිතතා සහ නීතිමය ප්‍රමිතීන්")
                refl = response.reflection_result
                if refl:
                    if refl.all_checks_passed:
                        st.markdown('<span class="badge-pass">✅ සියලුම කෘෂිකාර්මික සුරක්ෂිතතා සහ ප්‍රමිතීන් අනුමතයි</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-warn">⚠️ ඇතැම් රාජ්‍ය උපදෙස් සහ අවවාද හඳුනාගෙන ඇත</span>', unsafe_allow_html=True)
                    
                    st.markdown("#### සුරක්ෂිතතා පරීක්ෂණ වාර්තාව:")
                    for v in refl.verdicts:
                        icon = "✅" if v.passed else "⚠️"
                        st.write(f"{icon} **{v.check_name}**: {v.message}")
                    
                    st.markdown("#### අදාළ පනත් සහ දෙපාර්තමේන්තු නීති රීති:")
                    for cite in refl.regulatory_citations:
                        st.write(f"- 📜 {cite}")
                else:
                    st.write("සාමාන්‍ය තොරතුරු සඳහා වෙනම සුරක්ෂිතතා පරීක්ෂාවක් අවශ්‍ය නොවේ.")

            # TAB 4 (OPTIONAL VIVA DEV MODE)
            if viva_mode:
                with tabs[3]:
                    st.markdown("### 👨‍🏫 Viva Evaluator - Real-time Agent Communication Flow")
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

        except Exception as e:
            st.error(f"ක්‍රියාත්මක වීමේ දෝෂයකි: {e}")
            st.info("ඔබේ `.env` ගොනුවේ `GROQ_API_KEY` නිවැරදිව පවතීදැයි පරීක්ෂා කරන්න.")


# Footer
st.divider()
st.caption("Horizon Campus — Module IT41043: Agentic AI Assignment | Built for Sri Lankan Farmers with LangChain, FAISS, Groq & Streamlit")
