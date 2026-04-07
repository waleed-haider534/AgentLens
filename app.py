import streamlit as st
import pandas as pd
import time
from agent_core import get_llm_recommendations
from ollama_utils import get_local_models, test_local_model, is_ollama_running
from config import OLLAMA_MODEL

# Page config
st.set_page_config(
    page_title="AgentLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with cohesive friendly color scheme
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Color Variables - Friendly Teal/Cyan Scheme */
    :root {
        --primary: #0d9488;
        --primary-light: #14b8a6;
        --primary-dark: #0f766e;
        --secondary: #6366f1;
        --accent: #f59e0b;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --bg-main: #f0fdfa;
        --bg-card: #ffffff;
        --bg-sidebar: #fafafa;
        --text-main: #1e293b;
        --text-muted: #64748b;
        --border: #e2e8f0;
    }

    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Main background - soft gradient */
    .stApp {
        background: linear-gradient(180deg, #f0fdfa 0%, #ecfeff 50%, #f0f9ff 100%);
        min-height: 100vh;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }

    /* Main header with animation */
    .main-header {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 50%, #2dd4bf 100%);
        padding: 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(13, 148, 136, 0.3);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: shimmer 3s ease-in-out infinite;
    }

    @keyframes shimmer {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }

    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        position: relative;
    }

    .main-header p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.15rem !important;
        position: relative;
    }

    /* Input field styling - FRIENDLY & INVITING */
    .stTextArea > div > div > textarea {
        border-radius: 16px !important;
        border: 2px solid #ccfbf1 !important;
        padding: 1.2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%) !important;
        color: #1e293b !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 4px rgba(13, 148, 136, 0.15), 0 4px 12px rgba(13, 148, 136, 0.1) !important;
        background: #ffffff !important;
    }

    .stTextArea > div > div > textarea::placeholder {
        color: #94a3b8 !important;
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.8rem 1rem !important;
        background: #ffffff !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 4px rgba(13, 148, 136, 0.15) !important;
    }

    /* Primary button - ENHANCED */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.35) !important;
        text-transform: none !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(13, 148, 136, 0.45) !important;
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
    }

    /* Secondary button */
    .stButton > button:not([kind="primary"]) {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        color: #475569 !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:not([kind="primary"]):hover {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
        color: white !important;
        border-color: #0d9488 !important;
    }

    /* Card styling */
    .llm-card, .quick-action-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid #f1f5f9;
    }

    .llm-card:hover, .quick-action-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(13, 148, 136, 0.15);
        border-color: #ccfbf1;
    }

    /* Expander styling */
    .stExpander {
        border: none !important;
        border-radius: 20px !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.05) !important;
        overflow: hidden;
        margin-bottom: 1rem !important;
        background: #ffffff;
    }

    .stExpander > div > div {
        background: #ffffff !important;
        border-radius: 20px !important;
    }

    .stExpander > div > div > div[role="button"] {
        background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%) !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(13, 148, 136, 0.08);
        border: 1px solid #ccfbf1;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0d9488 !important;
        font-weight: 700 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #ffffff;
        padding: 0.5rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 0.7rem 1.3rem;
        font-weight: 500;
        color: #64748b;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
    }

    /* Feature tags - COLORFUL */
    .feature-tag {
        display: inline-block;
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        color: white;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
    }

    .feature-tag-free {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    }

    .feature-tag-paid {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
    }

    .feature-tag-tool {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0d9488 0%, #14b8a6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 14px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.3);
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%);
        border-left: 4px solid #0d9488;
        padding: 1.2rem;
        border-radius: 0 16px 16px 0;
        margin: 1rem 0;
    }

    .info-box-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left-color: #f59e0b;
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 16px !important;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(13, 148, 136, 0.1) !important;
    }

    /* Success/Error/Warning messages */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 12px !important;
    }

    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 12px !important;
    }

    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 12px !important;
    }

    /* Spinner */
    .stSpinner > div > div {
        border-color: #0d9488 !important;
        border-left-color: transparent !important;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
    }

    .empty-state-icon {
        font-size: 4.5rem;
        margin-bottom: 1rem;
        animation: bounce 2s ease-in-out infinite;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
        color: white !important;
        border-radius: 12px !important;
    }

    /* Sidebar elements */
    .sidebar-title {
        color: #0d9488;
        font-weight: 700;
    }

    /* Divider */
    hr, .stDivider {
        border-color: #e2e8f0 !important;
    }

    /* Quick suggestion buttons */
    .stButton > button:has(span:contains("...")) {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
        color: #0369a1 !important;
        border: 1px solid #bae6fd !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
    }

    /* Search suggestions styled as pills */
    div[data-testid="stHorizontalBlock"]:has(.stButton > button:contains("...")) {
        gap: 0.5rem;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem !important;
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "search_count" not in st.session_state:
    st.session_state.search_count = 0

# Sample queries for quick start
SAMPLE_QUERIES = [
    "I need an LLM for a customer support chatbot with tool calling",
    "Building a code generation agent for Python development",
    "Looking for a free model for text summarization tasks",
    "Need a model with large context window for document analysis",
    "Creating an agent that integrates with external APIs"
]

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    # Logo/Title
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem; animation: pulse 2s ease-in-out infinite;">
            🔍
        </div>
        <h2 style="color: #0d9488; margin: 0; font-weight: 700; font-size: 1.6rem;">AgentLens</h2>
        <p style="color: #64748b; font-size: 0.9rem; margin-top: 0.4rem;">Your LLM Discovery Assistant</p>
    </div>
    <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # System Status Section
    st.markdown("### ⚡ System Status")

    ollama_status = is_ollama_running()
    if ollama_status:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 0.9rem; border-radius: 14px; border: 1px solid #6ee7b7;">
            <span style="color: #059669; font-size: 1.3rem;">●</span>
            <span style="color: #059669; font-weight: 600;">Ollama Running</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 0.9rem; border-radius: 14px; border: 1px solid #fca5a5;">
            <span style="color: #dc2626; font-size: 1.3rem;">●</span>
            <span style="color: #dc2626; font-weight: 600;">Ollama Stopped</span>
        </div>
        <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.6rem;">Run <code style="background: #f1f5f9; padding: 0.2rem 0.5rem; border-radius: 6px;">ollama serve</code> to start</p>
        """, unsafe_allow_html=True)

    # Active model
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); padding: 1.1rem; border-radius: 14px; margin-top: 1rem; border: 1px solid #5eead4;">
        <div style="color: #64748b; font-size: 0.8rem; font-weight: 500;">🤖 Active Model</div>
        <div style="color: #0d9488; font-weight: 600; font-size: 1rem;">{OLLAMA_MODEL}</div>
    </div>
    """, unsafe_allow_html=True)

    # Local models count
    local_models = get_local_models()
    st.markdown(f"""
    <div style="display: flex; gap: 0.8rem; margin-top: 1rem;">
        <div style="flex: 1; background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%); padding: 1rem; border-radius: 14px; text-align: center; box-shadow: 0 2px 12px rgba(13, 148, 136, 0.1); border: 1px solid #ccfbf1;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #0d9488;">{len(local_models)}</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500;">Models</div>
        </div>
        <div style="flex: 1; background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%); padding: 1rem; border-radius: 14px; text-align: center; box-shadow: 0 2px 12px rgba(99, 102, 241, 0.1); border: 1px solid #c7d2fe;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #6366f1;">{st.session_state.search_count}</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500;">Searches</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Actions
    st.markdown("### ⚡ Quick Actions")

    col_qa1, col_qa2 = st.columns(2)
    with col_qa1:
        if st.button("🎲 Random", use_container_width=True):
            import random
            example = random.choice(SAMPLE_QUERIES)
            st.session_state.example_query = example
            st.rerun()

    with col_qa2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # Search History
    st.markdown("### 📜 Search History")
    if st.session_state.history:
        for idx, h in enumerate(st.session_state.history[-5:]):
            if st.button(f"🔎 {h}", key=f"history_{idx}", use_container_width=True):
                st.session_state.example_query = h
                st.rerun()
    else:
        st.markdown("""
        <p style="color: #94a3b8; text-align: center; padding: 1.2rem; background: #f8fafc; border-radius: 12px;">
            No searches yet.<br>Start exploring!
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # How it works
    with st.expander("📖 How it works", expanded=False):
        st.markdown("""
        <div style="line-height: 2.2; color: #475569;">
            <strong style="color: #0d9488;">1.</strong> Describe your use case<br>
            <strong style="color: #0d9488;">2.</strong> Get AI recommendations<br>
            <strong style="color: #0d9488;">3.</strong> Compare & choose<br>
            <strong style="color: #0d9488;">4.</strong> Test locally
        </div>
        """, unsafe_allow_html=True)

# ─── Main UI ───────────────────────────────────────────────
# Header
st.markdown("""
<div class="main-header" style="text-align: center;">
    <h1>🔍 AgentLens</h1>
    <p>Discover the perfect LLMs for your agentic AI workflow</p>
</div>
""", unsafe_allow_html=True)

# Query input section
st.markdown("### 📝 Describe Your Workflow")

# Handle example query from session state
if "example_query" in st.session_state:
    default_value = st.session_state.example_query
    del st.session_state.example_query
else:
    default_value = ""

query = st.text_area(
    "",
    value=default_value,
    placeholder="e.g. I'm building a marketing automation agent that handles campaign creation, audience targeting, and report generation. Recommend the best LLMs.",
    height=130,
    label_visibility="collapsed"
)

# Quick suggestion pills
st.markdown("**💡 Try these:**")
cols = st.columns(5)
for idx, sq in enumerate(SAMPLE_QUERIES):
    with cols[idx]:
        if st.button("✨ " + sq[:25], key=f"suggestion_{idx}", use_container_width=True):
            st.session_state.example_query = sq
            st.rerun()

# Action buttons
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    search_clicked = st.button("🚀 Search LLMs", type="primary", use_container_width=True)
with col_btn2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    query = ""

st.markdown("---")

# Results section
if search_clicked and query.strip():

    # Increment search count
    st.session_state.search_count += 1

    # Save to history
    history_entry = query[:50] + "..." if len(query) > 50 else query
    if history_entry not in st.session_state.history:
        st.session_state.history.append(history_entry)

    # Get recommendations
    with st.spinner("🤔 Analyzing your workflow and finding the best LLMs..."):
        time.sleep(0.5)
        recommendations = get_llm_recommendations(query)

    if not recommendations:
        st.error("❌ No recommendations returned. Try rephrasing your query or check if Ollama is running.")
    else:
        # Success message
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.5rem; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 1rem 1.5rem; border-radius: 14px; border-left: 4px solid #10b981;">
            <span style="font-size: 1.5rem;">🎉</span>
            <span style="font-size: 1.2rem; font-weight: 600; color: #059669;">
                Found {len(recommendations)} perfect matches!
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["🏆 Recommendations", "📊 Compare", "💻 Local Models"])

        with tab1:
            # LLM Cards
            for i, model in enumerate(recommendations):
                rank = i + 1

                # Build features tags
                features_html = ""
                for f in model.get("key_features", []):
                    features_html += f'<span class="feature-tag">{f}</span>'

                # Tool/function calling badge
                tool_support = model.get("tool_calling_support", "N/A")
                tool_class = "feature-tag-tool" if "yes" in tool_support.lower() else ""
                tool_icon = "✅" if "yes" in tool_support.lower() else "❌"

                # Cost badge
                cost = model.get("cost_tier", "N/A")
                cost_class = "feature-tag-free" if cost.lower() == "free" else "feature-tag-paid"

                with st.expander(f"#{rank} — {model.get('name', 'Unknown')}", expanded=(i == 0)):
                    col1, col2, col3 = st.columns([3, 2, 1])

                    with col1:
                        st.markdown(f"**📝 {model.get('name', 'Unknown')}")
                        st.markdown(f"<p style='color: #64748b;'>{model.get('description', 'N/A')}</p>", unsafe_allow_html=True)
                        st.markdown(f"**🏢 Provider:** {model.get('provider', 'N/A')}")

                    with col2:
                        st.markdown("**⚡ Key Features**")
                        st.markdown(features_html, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"**📦 Parameters:**<br>`{model.get('parameters', 'N/A')}`", unsafe_allow_html=True)
                        st.markdown(f"**🔧 Tools:** <span class='feature-tag {tool_class}'>{tool_icon} {tool_support}</span>", unsafe_allow_html=True)
                        st.markdown(f"**💰 Cost:** <span class='feature-tag {cost_class}'>{cost}</span>", unsafe_allow_html=True)

                    # Save button
                    col_fav, _ = st.columns([1, 3])
                    with col_fav:
                        fav_key = f"fav_{model.get('name', 'Unknown')}"
                        if st.button(f"❤️ Save", key=fav_key):
                            if model not in st.session_state.favorites:
                                st.session_state.favorites.append(model)
                            st.success(f"Saved!")

        with tab2:
            # Comparison Table
            table_data = []
            for i, model in enumerate(recommendations):
                tool_support = model.get("tool_calling_support", "")
                table_data.append({
                    "Rank": f"#{i + 1}",
                    "Model": model.get("name", "N/A"),
                    "Provider": model.get("provider", "N/A"),
                    "Params": model.get("parameters", "N/A"),
                    "Tools": "✅ Yes" if "yes" in tool_support.lower() else "❌ No",
                    "Cost": model.get("cost_tier", "N/A"),
                })

            df = pd.DataFrame(table_data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.TextColumn("Rank", width="small"),
                    "Model": st.column_config.TextColumn("Model Name", width="large"),
                    "Provider": st.column_config.TextColumn("Provider", width="medium"),
                    "Params": st.column_config.TextColumn("Parameters", width="small"),
                    "Tools": st.column_config.TextColumn("Tool Calling", width="medium"),
                    "Cost": st.column_config.TextColumn("Cost Tier", width="small"),
                }
            )

            # Download as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison",
                data=csv,
                file_name="llm_comparison.csv",
                mime="text/csv"
            )

        with tab3:
            # Local Models Section
            if local_models:
                for lm in local_models:
                    with st.expander(f"🖥️ {lm['name']}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Size", lm['size'])
                            st.metric("Parameters", lm['parameters'])

                        with col2:
                            st.metric("Family", lm['family'])
                            st.metric("Quantization", lm['quantization'])

                        # Test prompt section
                        st.markdown("---")
                        st.markdown("**🧪 Test this model**")

                        test_prompt = st.text_input(
                            "Enter a test prompt:",
                            key=f"test_{lm['name']}",
                            placeholder="e.g. Can you call tools?",
                            label_visibility="collapsed"
                        )

                        if st.button(f"🚀 Test {lm['name']}", key=f"btn_{lm['name']}"):
                            with st.spinner("Testing model..."):
                                result = test_local_model(lm["name"], test_prompt)
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%); padding: 1.5rem; border-radius: 14px; border-left: 4px solid #0d9488; margin-top: 1rem;">
                                <strong>🤖 Response:</strong><br><br>
                                {result}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box info-box-warning">
                    <strong>⚠️ No local Ollama models found.</strong><br>
                    Install models using: <code>ollama pull &lt;model-name&gt;</code>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("### Quick Install")
                st.code("ollama pull llama3.2\nollama pull mistral\nollama pull codellama", language="bash")

        # Saved favorites section
        if st.session_state.favorites:
            st.markdown("---")
            st.markdown("### ❤️ Your Saved Models")
            fav_cols = st.columns(min(len(st.session_state.favorites), 4))
            for idx, fav in enumerate(st.session_state.favorites):
                with fav_cols[idx % 4]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); padding: 1rem; border-radius: 14px; text-align: center; border: 1px solid #f9a8d4;">
                        <strong>{fav.get('name', 'Unknown')}</strong><br>
                        <span style="font-size: 0.8rem; color: #64748b;">{fav.get('provider', 'N/A')}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; padding: 1.5rem; background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%); border-radius: 14px;">
            <p>🤖 Powered by <strong style="color: #0d9488;">AgentLens</strong> — Built with Streamlit & Ollama</p>
        </div>
        """, unsafe_allow_html=True)

elif search_clicked and not query.strip():
    st.warning("⚠️ Please enter a query first!")

# Empty state when no search yet
elif not search_clicked:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <h3 style="color: #1e293b;">Ready to discover your perfect LLM?</h3>
        <p style="color: #64748b;">Describe your agentic workflow above and let AgentLens find the best models for you.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    st.markdown("### ✨ What AgentLens Can Do")
    cols = st.columns(4)

    features = [
        ("🤖", "Smart Recommendations", "AI-powered suggestions tailored to your use case"),
        ("📊", "Compare Models", "Side-by-side comparison of parameters, cost, and features"),
        ("💻", "Test Locally", "Try local Ollama models directly in the app"),
        ("💾", "Save Favorites", "Bookmark your favorite models for later")
    ]

    for idx, (icon, title, desc) in enumerate(features):
        with cols[idx]:
            st.markdown(f"""
            <div class="quick-action-card">
                <div style="font-size: 2.2rem;">{icon}</div>
                <h3 style="color: #1e293b; margin: 0.5rem 0;">{title}</h3>
                <p style="color: #64748b; font-size: 0.85rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)