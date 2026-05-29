# Customer Service AI Agent - Assignment 3

## Author
* Anna Chudilin

## Features implemented
* **Core Agent**: Built with LangGraph ReAct workflow, custom tool-calling, and Llama 3.3-70B model.
* **Custom Checkpointer**: Fault-tolerant `SqliteCheckpointSaver` managing ACID transactions across steps.
* **Bonus A (+10 pts)**: Streamlit chat interface showing reasoning steps (`st.expander` for tool calls & inputs).
* **Bonus B (+10 pts)**: Interactive query recommender based on episodic memory with deferred query execution.


# Customer Service Data Analyst Agent

## 1. Setup

1. Create a Python 3.13 virtual environment and activate it:
```bash
python3.13 -m venv .ai3
source .ai3/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Provide the Bitext dataset CSV at `data/bitext.csv`, or pass a custom dataset path with `--dataset`. 
Download the dataset from here: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
4. Configure your Nebius/OpenAI-compatible credentials:
```bash
export NEBIUS_API_KEY="your_api_key"
export NEBIUS_API_BASE="https://api.tokenfactory.nebius.com/v1/"
export NEBIUS_MODEL="meta-llama/Llama-3.3-70B-Instruct"
```

## 2. Run Streamlit Web Application

```bash
streamlit run app_streamlit.py
```

## 3. Run CLI

```bash
python3.13 main.py --session anna
```

## 4. Run MCP Server

```bash
python3.13 -m app.msp.server
```

## 5. Architecture Overview

- **Model selection**: The agent uses `langchain_openai.ChatOpenAI` backed by the `Meta-Llama-3.1-8B-Instruct` model hosted on the Nebius Token Factory endpoint. This open-weight model provides excellent instruction-following capabilities for structured text parsing and reasoning tasks.
- **Custom ReAct Loop**: Due to target API constraints regarding native schema serialization (`chat_template` payload errors), the architecture leverages a custom text-based ReAct loop orchestrated via a LangGraph `StateGraph`. The model generates explicit markdown JSON blocks (`{ "action": "tool_name", ... }`) which are captured, parsed, and routed to Python execution blocks via conditional edges in the state machine.
- **Query Routing**: An explicit `classify_query` preprocessing step functions as a hard constraint router node. It classifies incoming queries into `structured`, `unstructured`, or `out_of_scope` categories, ensuring the LLM gracefully declines out-of-scope prompts without hallucinating from its generic background weights.
- **Tools**: The project defines explicit, strictly typed tools equipped with comprehensive functional docstrings and validation rules:
  - `list_categories` — Fetch all text categories.
  - `list_intents` — List available customer intents.
  - `count_by_intent` — Count records for a specific intent.
  - `intents_by_category` — Get the distribution of intents inside one category.
  - `list_examples` — Show dataset examples.
  - `intent_distribution` — Get statistics on overall intent distribution.
  - `summarize_category` — Summarize data for a category.
  - `calculate_expression` — Evaluate basic arithmetic expressions.
- **Thread-Safe SQLite Checkpointing**: Persistent state management is handled by a custom `SqliteCheckpointSaver` back-end pointing to `checkpoints/checkpoints.db`. It implements isolated transactional workflows (`PRAGMA journal_mode=WAL`) protected by concurrent Python `threading.Lock` primitives to prevent race conditions during parallel graph processing.

## 6. Automated Tests

Run the automated test suite from the workspace root after activating the virtual environment:

```bash
python3.13 -m unittest discover -s tests
```

This suite validates the router classification logic, the dataset tool functions, and expression evaluation.

## 7. Example Queries

*Structured* - questions with concrete, data-driven answers:
- `What categories exist in the dataset?`
- `How many refund requests did we get?`
- `Show me 5 examples from the SHIPPING category.`
- `What is the distribution of intents in the ACCOUNT category?`

*Unstructured* - open-ended questions requiring summarization: 
- `Summarize the FEEDBACK category.` 
- `How do customer service representatives typically respond to cancellation requests?`

*Out-of-scope* - questions unrelated to the dataset: 
- `Who won the 2024 Champions League?`
- `Write me a poem about customer service.`
- `What's the best CRM software for handling complaints?`
- `Who is the president of France?`

## 8. MCP Client Connection Example

The FastMCP server runs on a secure standard input/output (Stdio) transport layer protocol. To inspect or connect an external host client to the server tools, you can use the official `@modelcontextprotocol/inspector` tool:

```bash
# Install the universal MCP inspector tool globally
npm install -g @modelcontextprotocol/inspector

# Run the inspector pointed directly at your local Python server script
npx @modelcontextprotocol/inspector python3.13 -m app.msp.server
```

Alternatively, you can test it programmatically inside an active Python environment using the `mcp-cli` wrapper:

```bash
pip install mcp-cli
mcp-cli run python3.13 -m app.msp.server
```
