# Graph construction for the ReAct agent workflow using StateGraph.
# This module defines the graph structure for the agent's reasoning process, including the nodes for LLM invocation and tool execution, as well as the edge routing logic that determines when to call tools based on the LLM's output.
# The graph is built using the StateGraph framework, which allows for flexible orchestration of the agent's interactions with the LLM and the tools, while maintaining a clear separation of concerns between reasoning and action.
# The route_or_execute_tool function serves as the core routing mechanism, parsing the LLM's output for tool execution instructions and enforcing a maximum iteration limit to prevent infinite loops in reasoning.
# The execute_tool_node function handles the execution of the identified tools and updates the state accordingly before returning control back to the LLM for further reasoning or final response generation.
# The compile_react_workflow function compiles the entire graph into an executable workflow that can be invoked by the main application loop.

# app/graph.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import END, StateGraph, MessagesState

# Configuration constants
MAX_ITERATIONS = 12

# Prompt-based ReAct system prompt inside graph context.
# Strictly forces pure JSON formatting structures.
DEFAULT_SYSTEM_PROMPT = (
    "You are a customer service dataset analyst for the Bitext CSV. "
    "Answer only from dataset facts and tools. Do not hallucinate or answer general knowledge questions. "
    "Use the router classification to decide whether the question is structured, unstructured, or out of scope. "
    "If the user asks what you remember about them, answer from the session profile. "
    "If you cannot complete the task after 12 reasoning steps, stop and return a graceful fallback response.\n\n"

    "CRITICAL RULES FOR INTENTS AND CATEGORIES:\n"
    "- If the user asks for 'examples' or 'samples' along with a specific number (e.g., 'Show me 3 examples from the REFUND category'), "
    "you MUST call 'list_examples' and pass both the mapped category string and the exact requested integer into the 'count' argument "
    "(e.g., action_input={\"category\": \"REFUND\", \"count\": 3}).\n"
    "- If the user asks for general 'examples', 'samples', or text rows (e.g., 'Show me examples of people wanting their money back'), "
    "you MUST call the 'list_examples' tool. Map any conversational synonyms to the closest dataset category "
    "(e.g., 'money back' or 'wanting money back' maps strictly to category='REFUND', 'cancellation' maps to category='CANCEL').\n"
    "- If the user asks to 'summarize' a category or asks for an overview/explanation of a category (e.g., 'Summarize the FEEDBACK category'), "
    "you MUST strictly call the 'summarize_category' tool with the argument {\"category\": \"FEEDBACK\"} (or the requested category name in uppercase).\n"
    "- NEVER call 'count_by_intent' with a 'category' argument. 'count_by_intent' ONLY accepts the 'intent' argument.\n"
    "- If the user asks what categories exist, you MUST call 'list_categories'. NEVER call 'count_by_intent' with intent='categories'.\n"
    "- If the user asks what intents exist, you MUST call 'list_intents'.\n"
    "- When calling 'count_by_intent', you can pass either a general keyword or an exact technical label in lowercase.\n"
    "- Available explicit refund intents in this dataset are: 'get_refund', 'track_refund', and 'check_refund_policy'.\n"
    "- If the user asks a general question about 'refunds', the system maps it to 'get_refund' automatically. "
    "However, if the user explicitly names a specific intent (e.g., 'track refund' or 'refund policy'), "
    "you MUST pass that exact string (e.g., 'track_refund' or 'check_refund_policy') to the tool input argument.\n"
    "- NEVER generate an action named 'out_of_scope'. If a question reaches you, it has already been cleared by the router.\n\n"

    "MATHEMATICAL AND AGGREGATION RULES:\n"
    "- Use the 'calculate_expression' tool ONLY when you need to sum, add, subtract, multiply, or divide multiple numeric values that already exist in the conversation history.\n"
    "- The 'expression' argument MUST contain ONLY pure numbers and mathematical operators (e.g., \"120 + 50\", \"250 - 30\").\n"
    "- CRITICAL: NEVER embed function names, code, or other tool names inside the 'calculate_expression' input payload.\n"
    "- If the user asks for a single count, DO NOT use 'calculate_expression'. Just call 'count_by_intent' and return the result directly to the user.\n"
    "- CRITICAL GENERATION RULE: You MUST generate exactly ONE JSON tool block at a time. Never chain multiple tool blocks. Wait for the tool response.\n\n"

    "OUTPUT FORMAT RULES:\n"
    "Your entire response MUST be a single, valid JSON object. Do not include any conversational preamble, markdown fences like ```json, or explanations before or after the JSON. "
    "If you need to call a tool, output this format:\n"
    "{\n"
    '  "action": "tool_name",\n'
    '  "action_input": {"param_name": "value"}\n'
    "}\n"
    "Example for counting: {\"action\": \"count_by_intent\", \"action_input\": {\"intent\": \"refund\"}}\n\n"
    "If you have gathered enough information and are ready to provide the final answer to the user, output this format:\n"
    "{\n"
    '  "final_answer": "Your complete textual summary or answer here based on tool facts."\n'
    "}\n\n"
    "Available tools:\n"
    "- list_categories: Get all text categories. Takes no arguments.\n"
    "- list_intents: List available customer intents. Takes no arguments.\n"
    "- count_by_intent: Count records for a specific intent. Argument: intent (str, must be lowercase).\n"
    "- intents_by_category: Get the distribution of intents for a specific category. Argument: category (str).\n"
    "- list_examples: Show text sample rows from the dataset. Arguments: category (str, optional), count (int, optional, how many samples to return).\n"
    "- intent_distribution: Get statistics on intent distribution. Takes no arguments.\n"
    "- summarize_category: Summarize data for a category. Argument: category (str).\n"
    "- calculate_expression: Evaluate a string with numbers and basic math operators (+, -, *, /). Argument: expression (str).\n\n"
    "Begin your JSON output now."
)

# Global mapping layout to bind tool tokens to their respective functions
TOOLS_MAPPING: Dict[str, Any] = {}


def route_or_execute_tool(state: MessagesState) -> Literal["execute_tool", "end"]:
    """
    Edge routing logic. Parses the LLM's text output for tool execution JSON blocks.
    Enforces a hard ceiling of 12 reasoning iterations to prevent infinite loops.
    """
    messages = state["messages"]

    # Calculate current iteration count based on previous tool execution blocks
    tool_turns = [m for m in messages if isinstance(
        m, HumanMessage) and m.content and "response output:" in str(m.content)]
    if len(tool_turns) >= MAX_ITERATIONS:
        return "end"

    last_message = messages[-1]
    if isinstance(last_message, list):
        last_message = last_message[-1]

    content = getattr(last_message, "content", "")
    if isinstance(content, list):
        content = " ".join([str(item.get("text", "")) if isinstance(
            item, dict) else str(item) for item in content])
    content = str(content).strip()

    # If the LLM output is a JSON final answer block, terminate the workflow loop.
    if '"final_answer"' in content:
        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            parsed = json.loads(content[start_idx:end_idx])
            last_message.content = parsed["final_answer"]
        except Exception as e:
            print(
                f"[red]Error parsing assistant response JSON:[/red] {str(e)}. Original content: {content}"
            )
        return "end"

    # If the model requested a tool execution via explicit JSON action payload, route to the action node.
    if '"action"' in content and '"action_input"' in content:
        return "execute_tool"

    # Default fallback: end the graph if no actionable JSON was found.
    return "end"


def _normalize_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    content = getattr(raw, "content", "")
    if isinstance(content, list):
        content = " ".join([
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ])
    return str(content).strip()


def _parse_json_object(content: str) -> dict | None:
    if "{" not in content:
        return None

    start = content.find("{")
    brace_count = 0
    end = -1
    for idx in range(start, len(content)):
        if content[idx] == "{":
            brace_count += 1
        elif content[idx] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = idx + 1
                break
    if end == -1:
        return None

    json_text = content[start:end].strip()
    try:
        return json.loads(json_text)
    except Exception:
        return None


def _execute_tool_function(tool_func: Any, action_input: Any) -> Any:
    if isinstance(action_input, dict):
        return tool_func.invoke(action_input)
    if action_input:
        return tool_func.invoke(action_input)
    return tool_func.invoke({})


def _format_examples_result(ex_list: list[Any]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(ex_list, 1):
        if isinstance(item, dict):
            txt = item.get("response") or item.get("instruction") or item.get(
                "utterance") or item.get("text") or str(item)
            lines.append(f"{idx}. {str(txt).strip()}")
        elif isinstance(item, list):
            lines.append(f"{idx}. {' '.join([str(x) for x in item])}")
        else:
            lines.append(f"{idx}. {str(item).strip()}")
    return "\n".join(lines)


def _format_tool_result(result: Any) -> str:
    if isinstance(result, str):
        stripped = result.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except Exception:
                return stripped
        else:
            return result
    else:
        parsed = result

    if isinstance(parsed, dict) and "examples" in parsed:
        ex_list = parsed["examples"]
        if isinstance(ex_list, list):
            return _format_examples_result(ex_list)
        return str(ex_list)
    if isinstance(parsed, list):
        return "\n".join([f"{i}. {str(x).strip()}" for i, x in enumerate(parsed, 1)])
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False)
    return str(parsed)


def _tool_guidance(action: str) -> str:
    if action == "list_examples":
        return "Format your response strictly as a clean numbered list showing the exact example text snippets or instructions provided. Do NOT summarize or rephrase them."
    return "Please analyze this data and provide a clear final answer to the user."


def _sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    sanitized: list[BaseMessage] = []
    for msg in messages:
        msg_copy = msg.copy() if hasattr(msg, "copy") else msg
        if hasattr(msg_copy, "content"):
            msg_copy.content = _normalize_content(msg_copy.content)
        sanitized.append(msg_copy)
    return sanitized


def _extract_final_answer(response: Any, fallback: str) -> Any:
    if isinstance(response, list):
        response = response[-1]
    content = _normalize_content(response)
    if "final_answer" in content and "{" in content:
        parsed = _parse_json_object(content)
        if parsed and "final_answer" in parsed:
            response.content = parsed["final_answer"]
    if not getattr(response, "content", None) or str(response.content).strip() in ["{{Currency Symbol}}", ""]:
        response.content = fallback
    return response


def execute_tool_node(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    messages = state["messages"]
    last_message = messages[-1]
    content = _normalize_content(last_message)

    try:
        tool_call = _parse_json_object(content)
        if not tool_call:
            raise ValueError("No JSON tool call found.")

        action = tool_call.get("action")
        action_input = tool_call.get("action_input", {})
        if action not in TOOLS_MAPPING:
            return {"messages": [HumanMessage(content=f"Error: Tool '{action}' not found.")]}

        if action == "list_examples" and isinstance(action_input, dict):
            action_input.setdefault("session_id", "default")

        result = _execute_tool_function(TOOLS_MAPPING[action], action_input)
        clean_result_str = _format_tool_result(result)
        guidance = _tool_guidance(action)
        tool_message = HumanMessage(
            content=f"Tool '{action}' response output:\n{clean_result_str}\n\n{guidance}"
        )
        return {"messages": [tool_message]}
    except Exception as error:
        return {"messages": [HumanMessage(content=f"Error executing tool workflow loop: {str(error)}")]}


def compile_react_workflow(llm: Any, tools: List[Any], checkpointer: Any) -> Any:
    """
    Compiles the state machine graph using an inline synchronous execution loop.
    This guarantees that the Python tool code is executed immediately inside the node.
    """
    global TOOLS_MAPPING
    for tool in tools:
        TOOLS_MAPPING[tool.name] = tool

    def call_model(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        sanitized_state_messages = _sanitize_messages(state["messages"])
        response = llm.invoke(sanitized_state_messages)
        content = _normalize_content(response)
        tool_call = _parse_json_object(content)

        if tool_call and tool_call.get("action") in TOOLS_MAPPING:
            action = tool_call["action"]
            action_input = tool_call.get("action_input", {})
            result = _execute_tool_function(
                TOOLS_MAPPING[action], action_input)
            clean_result_str = _format_tool_result(result)
            guidance = _tool_guidance(action)
            tool_message = HumanMessage(
                content=f"Tool '{action}' response output:\n{clean_result_str}\n\n{guidance}"
            )
            final_history = _sanitize_messages(
                sanitized_state_messages + [response, tool_message])
            final_response = llm.invoke(final_history)
            final_response = _extract_final_answer(
                final_response, clean_result_str)
            return {"messages": [final_response]}

        if isinstance(response, list):
            response = response[-1]
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", execute_tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        route_or_execute_tool,
        {
            "execute_tool": "action",
            "end": END,
        }
    )
    workflow.add_edge("action", "agent")
    return workflow.compile(checkpointer=checkpointer)
