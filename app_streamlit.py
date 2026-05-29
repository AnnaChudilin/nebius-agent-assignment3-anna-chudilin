# Streamlit Web Application for AI Agent Interaction

import streamlit as st
from pathlib import Path
from uuid import uuid4
import traceback

# Import explicit message classes to ensure strict LangGraph state validation
from langchain_core.messages import HumanMessage, SystemMessage

# Import the builder function and default prompt from your repository modules
from app.agent import build_agent
from app.graph import DEFAULT_SYSTEM_PROMPT

# Configure main web application container layout
st.set_page_config(page_title="AI Agent - Assignment 3",
                   page_icon="🤖", layout="wide")

# -----------------------------------------------------------------------------
# PATH & ENVIRONMENT INITIALIZATION
# -----------------------------------------------------------------------------
possible_paths = ["data/bitext.csv",
                  "../data/bitext.csv", "app/data/bitext.csv"]
dataset_path = "data/bitext.csv"
for p in possible_paths:
    if Path(p).exists():
        dataset_path = p
        break

if "graph_instance" not in st.session_state:
    st.session_state.graph_instance = build_agent(dataset_path=dataset_path)

graph = st.session_state.graph_instance

# -----------------------------------------------------------------------------
# SIDEBAR: Session Management (Bonus A - Requirement 3)
# -----------------------------------------------------------------------------
st.sidebar.title("Configuration")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())[:8]

session_id_input = st.sidebar.text_input(
    "Conversation Session ID",
    value=st.session_state.session_id,
    help="Change this ID to switch to another thread or resume a past conversation."
)

if session_id_input != st.session_state.session_id:
    st.session_state.session_id = session_id_input
    if "messages" in st.session_state:
        del st.session_state.messages
    if "recommended_query" in st.session_state:
        del st.session_state.recommended_query
    if "awaiting_confirmation" in st.session_state:
        del st.session_state.awaiting_confirmation

st.sidebar.caption(f"Active Thread ID: **{st.session_state.session_id}**")

# -----------------------------------------------------------------------------
# MAIN UI: Chat Interface (Bonus A - Requirement 1)
# -----------------------------------------------------------------------------
st.title("🤖 Customer Service AI Agent")
st.caption("Nebius Academy - From AI Model to AI Agent (Assignment 3)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "recommended_query" not in st.session_state:
    st.session_state.recommended_query = None
if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if raw_user_query := st.chat_input("Ask a question or type 'What should I query next?'..."):
    clean_input = raw_user_query.lower().strip()

    with st.chat_message("user"):
        st.write(raw_user_query)
    st.session_state.messages.append(
        {"role": "user", "content": raw_user_query})

    config = {
        "configurable": {
            "thread_id": st.session_state.session_id,
            "checkpoint_ns": ""
        }
    }

    with st.chat_message("assistant"):
        final_answer = ""

        # BONUS B: STEP 4 - Confirmation Execution
        if st.session_state.awaiting_confirmation and clean_input in ["yes", "yes, do it", "do it", "confirm", "go ahead"]:
            query_to_run = st.session_state.recommended_query
            st.session_state.recommended_query = None
            st.session_state.awaiting_confirmation = False

            st.info(f"⚡ Executing confirmed query: *\"{query_to_run}\"*")
            input_state = {"messages": [HumanMessage(content=query_to_run)]}
            try:
                for chunk in graph.stream(input_state, config, stream_mode="updates"):
                    for node_name, node_output in chunk.items():
                        if not isinstance(node_output, dict):
                            continue
                        messages_list = node_output.get("messages", [])
                        if not messages_list:
                            continue
                        last_msg = messages_list[-1]

                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tool_call in last_msg.tool_calls:
                                with st.expander(f"🛠️ Tool Call: {tool_call['name']}", expanded=True):
                                    st.json(tool_call["args"])
                        elif getattr(last_msg, "type", None) == "tool":
                            with st.expander(f"📥 Tool Output from {last_msg.name}", expanded=False):
                                st.code(last_msg.content, language="json")
                        elif getattr(last_msg, "type", None) == "ai" and last_msg.content:
                            final_answer = last_msg.content

                if not final_answer:
                    state_now = graph.get_state(config)
                    if state_now and "messages" in state_now.values and state_now.values["messages"]:
                        final_answer = state_now.values["messages"][-1].content
            except Exception as graph_err:
                st.error(f"Graph execution failed: {str(graph_err)}")
                with st.expander("🔍 View Error Traceback"):
                    st.code(traceback.format_exc())

        # BONUS B: STEP 1, 2, 3 - Recommendation Logic
        elif "what should i query next" in clean_input or (st.session_state.awaiting_confirmation and "instead" in clean_input) or ("rather" in clean_input):
            history_context = ""
            state_now = graph.get_state(config)
            if state_now and "messages" in state_now.values:
                past_messages = state_now.values["messages"][-10:]
                history_context = " ".join(
                    [m.content for m in past_messages if getattr(m, "type", None) in ["human", "ai"]])

            recommender_prompt = (
                "You are an interactive query recommendation engine. "
                "Review the following conversation history and the user's latest request. "
                f"Conversation context from memory: {history_context if history_context else 'No queries executed yet.'} "
                f"User's latest request: '{raw_user_query}' "
                "Formulate a precise, specific follow-up query that can be executed via dataset tools (like count_by_intent or list_examples). "
                "Output rules:\n"
                "1. Suggest ONLY ONE specific text query inside double quotes (e.g., \"show 5 examples from the REFUND category\").\n"
                "2. Do NOT execute any tools yet.\n"
                "3. Explain why you recommend this query based on their profile interest and ask if you should execute it."
            )

            try:
                # FIX: Initialize an independent ChatOpenAI model using environment configurations
                # to completely bypass the CompiledStateGraph attribute limitation.
                from langchain_openai import ChatOpenAI
                import os

                model_name = os.getenv("NEBIUS_MODEL") or os.getenv(
                    "OPENAI_MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
                api_key = os.getenv("NEBIUS_API_KEY") or os.getenv(
                    "OPENAI_API_KEY")
                base_url = os.getenv("NEBIUS_API_BASE") or os.getenv(
                    "OPENAI_API_BASE") or "https://nebius.com"

                recommender_llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.2
                )

                res = recommender_llm.invoke([SystemMessage(
                    content=recommender_prompt), HumanMessage(content=raw_user_query)])
                final_answer = res.content

                # Dynamic safe fallback parser
                extracted_query = "show 5 examples from the ACCOUNT category"
                if '"' in final_answer:
                    parts = final_answer.split('"')
                    if len(parts) > 1:
                        extracted_query = parts[1]

                st.session_state.recommended_query = extracted_query
                st.session_state.awaiting_confirmation = True
            except Exception as prompt_err:
                st.error(
                    f"Failed to compile recommendation: {str(prompt_err)}")
                with st.expander("🔍 View Error Traceback"):
                    st.code(traceback.format_exc())

        # DEFAULT WORKFLOW: Direct Reactive Graph Execution Loop
        else:
            st.session_state.awaiting_confirmation = False
            st.session_state.recommended_query = None
            current_state = graph.get_state(config)

            if not current_state or not current_state.values or "messages" not in current_state.values or not current_state.values["messages"]:
                input_state = {"messages": [SystemMessage(
                    content=DEFAULT_SYSTEM_PROMPT), HumanMessage(content=raw_user_query)]}
            else:
                input_state = {"messages": [
                    HumanMessage(content=raw_user_query)]}

            try:
                for chunk in graph.stream(input_state, config, stream_mode="updates"):
                    for node_name, node_output in chunk.items():
                        if not isinstance(node_output, dict):
                            continue
                        messages_list = node_output.get("messages", [])
                        if not messages_list:
                            continue
                        last_msg = messages_list[-1]

                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tool_call in last_msg.tool_calls:
                                with st.expander(f"🛠️ Tool Call: {tool_call['name']}", expanded=True):
                                    st.json(tool_call["args"])
                        elif getattr(last_msg, "type", None) == "tool":
                            with st.expander(f"📥 Tool Output from {last_msg.name}", expanded=False):
                                st.code(last_msg.content, language="json")
                        elif getattr(last_msg, "type", None) == "ai" and last_msg.content:
                            final_answer = last_msg.content

                if not final_answer:
                    state_now = graph.get_state(config)
                    if state_now and "messages" in state_now.values and state_now.values["messages"]:
                        final_answer = state_now.values["messages"][-1].content
            except Exception as graph_err:
                st.error(f"Graph execution failed: {str(graph_err)}")
                with st.expander("🔍 View Error Traceback"):
                    st.code(traceback.format_exc())

        # MAIN RESPONSE RENDERING
        if final_answer:
            if st.session_state.awaiting_confirmation and st.session_state.recommended_query:
                st.markdown("### 💡 Recommendation Proposal")
            else:
                st.markdown("### Final Answer")
            st.write(final_answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer})
        else:
            st.warning(
                "Execution completed without returning text state nodes.")
