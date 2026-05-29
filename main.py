# Main application entry point for the Customer Service Data Analyst Agent.
# This script initializes the agent, loads the dataset, and manages the interactive session with the user.
# It incorporates the routing logic to classify user queries, updates the user profile based on interactions,
# and orchestrates the agent's reasoning and tool execution process.
# The main loop captures user input, processes it through the agent, and handles the streaming output
# to provide real-time feedback on the agent's reasoning steps and final responses.
# The application also manages session persistence through the ConversationMemory and UserProfile classes,
# allowing for a continuous and context-aware user experience across multiple interactions.

from __future__ import annotations

import argparse
import json
from typing import Any
from rich import print
from app.agent import DEFAULT_SYSTEM_PROMPT, build_agent
from app.memory import ConversationMemory
from app.profile import UserProfile
from app.router import classify_query
from app.dataset import set_active_session


def build_system_messages(profile_summary: str) -> list[tuple[str, str]]:
    """
    Construct system instructions dynamically matching current UserProfile context.
    """
    messages: list[tuple[str, str]] = [("system", DEFAULT_SYSTEM_PROMPT)]
    if profile_summary:
        messages.append(
            ("system", f"User profile facts (what you remember about them): {profile_summary}"))
    return messages


def _normalize_content(message_obj: Any) -> str:
    content = getattr(message_obj, "content", "")
    if isinstance(content, list):
        content = " ".join([
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ])
    return str(content).strip()


def _extract_tool_block(content: str) -> str | None:
    if "```json" in content and '"action"' in content:
        return content.split("```json")[-1].split("```")[0].strip()
    if content.startswith("{") and '"action"' in content:
        return content
    return None


def _handle_ai_message(content: str, printed_tool_calls: set[str]) -> str | None:
    tool_block = _extract_tool_block(content)
    if tool_block:
        if tool_block not in printed_tool_calls:
            print(
                f"[dim]Tool Request Block Generated:[/dim]\n[yellow]{tool_block}[/yellow]")
            printed_tool_calls.add(tool_block)
        return None
    return content


def _parse_assistant_json_content(assistant_content: str) -> str:
    if not assistant_content.startswith("{") or not assistant_content.endswith("}"):
        return assistant_content

    try:
        clean_data = json.loads(assistant_content)
        if "categories" in clean_data:
            return f"The categories available in the dataset are: {', '.join(clean_data['categories'])}."
        if "summary" in clean_data:
            return clean_data["summary"]
        if "count" in clean_data:
            return f"The total count is {clean_data['count']}."
    except Exception as error:
        print(
            f"[red]Error parsing assistant response JSON:[/red] {str(error)}. Original content: {assistant_content}"
        )

    return assistant_content


def _print_tool_feedback(content: str, printed_tool_calls: set[str]) -> None:
    feedback_key = f"exec_{content[:60]}"
    if feedback_key not in printed_tool_calls:
        print(
            f"[dim]Tool Execution Result Output:[/dim] [magenta]{content}[/magenta]")
        printed_tool_calls.add(feedback_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent")
    parser.add_argument("--session", default="default",
                        help="Session ID for persistent memory.")
    parser.add_argument("--dataset", default="data/bitext.csv",
                        help="Path to the Bitext dataset CSV file.")
    args = parser.parse_args()

    memory = ConversationMemory(args.session)
    profile = UserProfile(args.session)
    agent = build_agent(dataset_path=args.dataset)
    config = {"configurable": {"thread_id": args.session}}

    print("[green]Customer Service Analyst Agent[/green]")
    print("Type 'exit' or 'quit' to close the session.")

    while True:
        set_active_session(args.session)
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            break
        if not user_input:
            continue

        route = classify_query(user_input)
        print(f"[cyan]Router:[/cyan] {route.query_type} — {route.reason}")

        if route.query_type == "out_of_scope":
            print(
                "[yellow]Assistant:[/yellow] I'm sorry, I can only answer questions about the Bitext customer service dataset."
            )
            profile.update(user_input)
            continue

        profile.update(user_input)
        memory.append("user", user_input)

        system_updates = build_system_messages(profile.summary())
        input_payload = {"messages": system_updates + [("user", user_input)]}

        print("[cyan]Reasoning steps:[/cyan]")
        assistant_content = ""
        printed_tool_calls: set[str] = set()

        stream = agent.stream(
            input_payload,
            config=config,
            stream_mode="values",
        )

        for event in stream:
            if "messages" not in event or not event["messages"]:
                continue
            for message_obj in event["messages"]:
                if isinstance(message_obj, list):
                    if not message_obj:
                        continue
                    message_obj = message_obj[-1]

                content = _normalize_content(message_obj)
                if not content:
                    continue

                msg_type = getattr(message_obj, "type", "")
                if msg_type == "ai":
                    candidate = _handle_ai_message(content, printed_tool_calls)
                    if candidate:
                        assistant_content = candidate
                elif msg_type == "human" and "response output:" in content:
                    _print_tool_feedback(content, printed_tool_calls)

        print()
        if assistant_content:
            assistant_content = _parse_assistant_json_content(
                assistant_content)
            print(f"[green]Assistant:[/green] {assistant_content}")
            memory.append("assistant", assistant_content)
        else:
            print(
                "[red]Assistant:[/red] No final analysis response text was captured from the graph.")

    print("[green]Session ended.[/green]")


if __name__ == "__main__":
    main()
