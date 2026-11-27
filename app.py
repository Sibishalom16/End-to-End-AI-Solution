"""Study Planner AI Agent - Streamlit Application

A production-quality student study companion featuring:
- LangGraph intelligent multi-path routing (DOCS, TOOLS, DECLINE)
- Grounded RAG with source citations from knowledge_base/
- Custom deterministic calculation tools for study hours and study plans
- Conversation memory with LCEL-based query rewriting
- Secure per-visitor Gemini API key management (never hardcoded)
- Execution telemetry: route badges, execution paths, expandable citations & agent steps
"""

import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

import graph_core
from tools import calculate_study_hours, generate_study_plan

# ------------------ Page Configuration ------------------ #

st.set_page_config(
    page_title="Study Planner AI Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom Styling ------------------ #

st.markdown("""
<style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Header typography */
    .app-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* Route & Pipeline Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
        align-items: center;
    }
    .route-badge-docs {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .route-badge-tools {
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .route-badge-decline {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .path-badge {
        background-color: #F3F4F6;
        color: #374151;
        border: 1px solid #E5E7EB;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-family: monospace;
    }
    .rewrite-note {
        font-size: 0.8rem;
        color: #6B7280;
        font-style: italic;
        margin-bottom: 0.5rem;
    }

    /* Sidebar cards */
    .sidebar-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------ Session State Initialization ------------------ #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ------------------ Sidebar: Credentials & Controls ------------------ #

with st.sidebar:
    st.title("🎓 Study Planner Settings")
    st.markdown("Configure your AI session parameters below.")

    # 1. API Key Input
    env_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    api_key = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        placeholder="AIzaSy...",
        help="Enter your Gemini API key from Google AI Studio. Your key is stored in session memory only and never written to disk or transmitted to external servers."
    )

    if api_key:
        st.success("API Key Active (Ready)", icon="✅")
    else:
        st.warning("Please enter your Gemini API key to activate the agent.", icon="🔑")

    # 2. Model Selection
    model_name = st.selectbox(
        "Gemini Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        index=0,
        help="gemini-1.5-flash is fast, economical, and recommended for conversational study queries."
    )

    st.markdown("---")

    # 3. Memory & Chat Management
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")

    # 4. Knowledge Base Reference
    with st.expander("📚 Knowledge Base Status", expanded=False):
        st.markdown("""
        **Indexed Study Documents:**
        - `study_techniques.md`: Pomodoro, Active Recall, Spaced Repetition, Leitner Box, Feynman Technique, Interleaving, SQ3R.
        - `time_management.md`: Eisenhower Matrix, Pareto 80/20, Time-Blocking, Eat That Frog, Procrastination protocols.
        - `exam_preparation.md`: 4-Week Countdown, Traffic Light Audit, Timed Practice Exams, Error Logs, Cornell Notes.
        """)

    with st.expander("📐 Architecture & Routing", expanded=False):
        st.markdown("""
        **LangGraph Routing Logic:**
        - **DOCS**: Academic study theory and techniques grounded in local documents.
        - **TOOLS**: Deterministic hour calculations and structured study schedule synthesis.
        - **DECLINE**: Scope guard for non-academic queries or prompt injection attacks.
        """)


# ------------------ Main Application Header ------------------ #

st.markdown('<div class="app-header">🎓 Study Planner AI Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Your intelligent academic companion. Ask conceptual questions about study methods, '
    'calculate study hours, or generate structured study schedules with strict grounding and safety.'
    '</div>',
    unsafe_allow_html=True
)


# ------------------ Extra Feature: Suggested Questions ------------------ #

st.markdown("##### 💡 Suggested Questions (Click to Ask):")
sug_col1, sug_col2, sug_col3, sug_col4 = st.columns(4)

with sug_col1:
    if st.button("📖 Explain the Pomodoro technique", use_container_width=True):
        st.session_state.pending_prompt = "Explain the Pomodoro technique and its recommended intervals."

with sug_col2:
    if st.button("🧠 How does active recall work?", use_container_width=True):
        st.session_state.pending_prompt = "What is active recall and how does the Feynman technique help?"

with sug_col3:
    if st.button("📅 Create a 7-day study plan", use_container_width=True):
        st.session_state.pending_prompt = "Create a 7-day study plan for Calculus, Physics, and Chemistry studying 4 hours per day."

with sug_col4:
    if st.button("⏱️ Calculate study hours (14 days)", use_container_width=True):
        st.session_state.pending_prompt = "Calculate my study hours for 14 available days at 3.5 hours per day."

st.markdown("---")


# ------------------ Chat History Rendering ------------------ #

for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")

    with st.chat_message(role):
        if role == "assistant":
            route = msg.get("route", "")
            execution_path = msg.get("execution_path", [])
            sources = msg.get("sources", [])
            agent_steps = msg.get("agent_steps", [])
            rewritten_q = msg.get("rewritten_question", "")
            original_q = msg.get("original_question", "")

            # Display Telemetry Badges
            badges_html = '<div class="badge-container">'
            if route == "DOCS":
                badges_html += '<span class="route-badge-docs">📚 Route: DOCS</span>'
            elif route == "TOOLS":
                badges_html += '<span class="route-badge-tools">⚙️ Route: TOOLS</span>'
            elif route == "DECLINE":
                badges_html += '<span class="route-badge-decline">🛡️ Route: DECLINE</span>'
            
            if execution_path:
                path_str = " ➔ ".join(execution_path)
                badges_html += f'<span class="path-badge">Path: {path_str}</span>'
            badges_html += '</div>'
            st.markdown(badges_html, unsafe_allow_html=True)

            # Display follow-up rewrite info if applicable
            if rewritten_q and original_q and rewritten_q.strip().lower() != original_q.strip().lower():
                st.markdown(f'<div class="rewrite-note">🔄 Standalone Query: <em>"{rewritten_q}"</em></div>', unsafe_allow_html=True)

            # Render Main Markdown Content
            st.markdown(content)

            # Expandable Sources (DOCS route)
            if sources:
                with st.expander("📚 Source Documents Cited", expanded=False):
                    for src in sources:
                        st.markdown(f"- 📄 `{src}`")

            # Expandable Agent Steps (TOOLS route)
            if agent_steps:
                with st.expander("🛠️ Agent Execution Steps & Calculations", expanded=False):
                    for step in agent_steps:
                        st.markdown(f"- {step}")
        else:
            # User Message
            st.markdown(content)


# ------------------ User Input & Processing ------------------ #

# Check if a suggested question was clicked
active_prompt = None
if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    active_prompt = st.chat_input("Ask a study question, calculate hours, or generate a study plan...")

if active_prompt:
    if not api_key:
        st.error("⚠️ **Google Gemini API Key Required**: Please enter your API key in the sidebar before asking questions.")
    else:
        # 1. Render and record User Message
        st.session_state.messages.append({
            "role": "user",
            "content": active_prompt
        })
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # 2. Execute LangGraph Workflow
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing via LangGraph (Rewriting ➔ Routing ➔ Execution)..."):
                result = graph_core.run_study_graph(
                    question=active_prompt,
                    chat_history=st.session_state.chat_history,
                    api_key=api_key,
                    model_name=model_name
                )

            route = result.get("route", "DECLINE")
            execution_path = result.get("execution_path", [])
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            agent_steps = result.get("agent_steps", [])
            rewritten_q = result.get("rewritten_question", "")

            # Display Badges
            badges_html = '<div class="badge-container">'
            if route == "DOCS":
                badges_html += '<span class="route-badge-docs">📚 Route: DOCS</span>'
            elif route == "TOOLS":
                badges_html += '<span class="route-badge-tools">⚙️ Route: TOOLS</span>'
            elif route == "DECLINE":
                badges_html += '<span class="route-badge-decline">🛡️ Route: DECLINE</span>'
            
            if execution_path:
                path_str = " ➔ ".join(execution_path)
                badges_html += f'<span class="path-badge">Path: {path_str}</span>'
            badges_html += '</div>'
            st.markdown(badges_html, unsafe_allow_html=True)

            if rewritten_q and rewritten_q.strip().lower() != active_prompt.strip().lower():
                st.markdown(f'<div class="rewrite-note">🔄 Standalone Query: <em>"{rewritten_q}"</em></div>', unsafe_allow_html=True)

            # Display Answer
            st.markdown(answer)

            # Expandable Sources
            if sources:
                with st.expander("📚 Source Documents Cited", expanded=False):
                    for src in sources:
                        st.markdown(f"- 📄 `{src}`")

            # Expandable Agent Steps
            if agent_steps:
                with st.expander("🛠️ Agent Execution Steps & Calculations", expanded=False):
                    for step in agent_steps:
                        st.markdown(f"- {step}")

        # 3. Update Conversation History & Memory
        st.session_state.chat_history.append(HumanMessage(content=active_prompt))
        st.session_state.chat_history.append(AIMessage(content=answer))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "route": route,
            "execution_path": execution_path,
            "sources": sources,
            "agent_steps": agent_steps,
            "original_question": active_prompt,
            "rewritten_question": rewritten_q
        })
