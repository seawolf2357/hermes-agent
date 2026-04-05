"""
Hermes Agent - Data Analysis Demo for Hugging Face Spaces.

Provides a Gradio web UI with:
1. AI Chat tab - converse with Kimi K2.5 via Fireworks AI
2. Data Analysis tab - upload CSV/JSON, ask questions, get charts & stats
"""

import io
import json
import os
import re
import uuid
import traceback

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import requests as http_requests

# ---------------------------------------------------------------------------
# Configuration - Fireworks AI with Kimi K2.5
# ---------------------------------------------------------------------------
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2p5")

SYSTEM_PROMPT = """You are Hermes, an expert data analyst AI assistant built by Nous Research.
You help users analyze data, create visualizations, and extract insights.

When analyzing data, you MUST respond with executable Python code wrapped in ```python blocks.
The code should:
- Use the variable `df` which contains the uploaded pandas DataFrame
- Use matplotlib or plotly for visualizations
- Print results using print()
- Save any matplotlib figures to 'output.png' using plt.savefig('output.png', dpi=150, bbox_inches='tight')
- Be self-contained and runnable

When chatting without data, respond naturally and helpfully."""

DATA_ANALYSIS_PROMPT = """You are Hermes, an expert data analyst.
The user has uploaded a dataset. Here is the data summary:

{summary}

First 5 rows:
{head}

Column types:
{dtypes}

The user's question: {question}

Respond with:
1. A brief explanation of your analysis approach
2. Python code in a ```python block that:
   - Uses the pre-loaded `df` DataFrame
   - Answers the question with analysis/visualization
   - Prints key findings with print()
   - If creating a plot, saves it to 'output.png' using plt.savefig('output.png', dpi=150, bbox_inches='tight')
   - Uses plt.close() after saving"""


def fireworks_chat(messages: list[dict], stream: bool = False):
    """Call Fireworks AI API with the given messages."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": messages,
        "stream": stream,
    }
    return http_requests.post(
        FIREWORKS_URL,
        headers=headers,
        json=payload,
        stream=stream,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Chat Tab
# ---------------------------------------------------------------------------
def chat_respond(message: str, history: list[dict], session_id: str):
    """Stream a chat response from Fireworks AI (Kimi K2.5)."""
    if not FIREWORKS_API_KEY:
        yield "Please set the `FIREWORKS_API_KEY` environment variable in your Space settings."
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": message})

    try:
        resp = fireworks_chat(messages, stream=True)
        resp.raise_for_status()
        partial = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                partial += delta
                yield partial
    except Exception as e:
        yield f"Error: {e}"


# ---------------------------------------------------------------------------
# Data Analysis Tab
# ---------------------------------------------------------------------------
def load_data(file) -> tuple[pd.DataFrame | None, str]:
    """Load a CSV or JSON file into a DataFrame."""
    if file is None:
        return None, "No file uploaded."
    try:
        path = file.name if hasattr(file, "name") else str(file)
        if path.endswith(".json"):
            df = pd.read_json(path)
        else:
            df = pd.read_csv(path)
        summary = (
            f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}\n\n"
            f"**Columns:** {', '.join(df.columns.tolist())}"
        )
        return df, summary
    except Exception as e:
        return None, f"Failed to load file: {e}"


def get_data_summary(df: pd.DataFrame) -> str:
    """Generate a text summary of a DataFrame for the LLM prompt."""
    buf = io.StringIO()
    df.describe(include="all").to_string(buf)
    return buf.getvalue()


def extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""


def execute_analysis(code: str, df: pd.DataFrame) -> tuple[str, str | None]:
    """Execute analysis code safely and return output + optional image path."""
    output_capture = io.StringIO()
    image_path = None

    # Clean up any previous output
    if os.path.exists("output.png"):
        os.remove("output.png")

    local_vars = {"df": df.copy(), "pd": pd, "plt": plt, "px": px}

    try:
        exec_globals = {"__builtins__": __builtins__}
        exec_globals.update(local_vars)

        # Redirect print to capture
        import contextlib
        with contextlib.redirect_stdout(output_capture):
            exec(code, exec_globals)

        if os.path.exists("output.png"):
            image_path = "output.png"

    except Exception:
        output_capture.write(f"\nExecution Error:\n{traceback.format_exc()}")

    plt.close("all")
    return output_capture.getvalue(), image_path


def analyze_data(
    file, question: str, history: list[dict]
) -> tuple[list[dict], str, str | None, str]:
    """Main analysis pipeline: upload data, ask question, get results."""
    if not FIREWORKS_API_KEY:
        msg = "Please set the `FIREWORKS_API_KEY` environment variable."
        return history + [{"role": "assistant", "content": msg}], "", None, ""

    df, summary_md = load_data(file)
    if df is None:
        return (
            history + [{"role": "assistant", "content": summary_md}],
            "",
            None,
            "",
        )

    if not question.strip():
        return (
            history
            + [
                {
                    "role": "assistant",
                    "content": f"Data loaded successfully!\n\n{summary_md}\n\nAsk me a question about this data.",
                }
            ],
            "",
            None,
            "",
        )

    # Build prompt with data context
    prompt = DATA_ANALYSIS_PROMPT.format(
        summary=get_data_summary(df),
        head=df.head().to_string(),
        dtypes=df.dtypes.to_string(),
        question=question,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = fireworks_chat(messages, stream=False)
        resp.raise_for_status()
        result = resp.json()
        answer = result["choices"][0]["message"]["content"] or ""
    except Exception as e:
        answer = f"API Error: {e}"
        return (
            history
            + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            "",
            None,
            "",
        )

    # Extract and execute code
    code = extract_code(answer)
    output_text = ""
    image_path = None

    if code:
        output_text, image_path = execute_analysis(code, df)

    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    return updated_history, output_text, image_path, code


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def create_app():
    session_id = str(uuid.uuid4())

    with gr.Blocks(
        title="Hermes Agent - Data Analysis",
        theme=gr.themes.Soft(primary_hue="purple"),
    ) as demo:
        gr.Markdown(
            "# Hermes Agent - Data Analysis Demo\n"
            "An interactive demo of [Hermes Agent](https://github.com/NousResearch/hermes-agent) "
            "by Nous Research."
        )

        with gr.Tabs():
            # --- Chat Tab ---
            with gr.Tab("AI Chat"):
                gr.ChatInterface(
                    fn=chat_respond,
                    type="messages",
                    additional_inputs=[
                        gr.State(value=session_id),
                    ],
                    title="Chat with Hermes",
                    description="Ask anything - general questions, coding help, or data analysis guidance.",
                    examples=[
                        "What kinds of data analysis can you help me with?",
                        "Explain the difference between correlation and causation.",
                        "Write Python code to generate a sample dataset with pandas.",
                    ],
                )

            # --- Data Analysis Tab ---
            with gr.Tab("Data Analysis"):
                gr.Markdown(
                    "Upload a CSV or JSON file and ask questions about your data. "
                    "The agent will analyze it and generate visualizations."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Upload CSV or JSON",
                            file_types=[".csv", ".json"],
                        )
                        data_summary = gr.Markdown(label="Data Summary")
                        question_input = gr.Textbox(
                            label="Ask a question about your data",
                            placeholder="e.g., Show the distribution of values in column X",
                            lines=2,
                        )
                        analyze_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Analysis Conversation",
                            type="messages",
                            height=300,
                        )
                        with gr.Row():
                            with gr.Column():
                                output_text = gr.Textbox(
                                    label="Execution Output",
                                    lines=8,
                                    interactive=False,
                                )
                            with gr.Column():
                                output_image = gr.Image(
                                    label="Visualization",
                                    type="filepath",
                                )
                        code_display = gr.Code(
                            label="Generated Code",
                            language="python",
                            interactive=False,
                        )

                # Wire up data loading preview
                file_input.change(
                    fn=lambda f: load_data(f)[1],
                    inputs=[file_input],
                    outputs=[data_summary],
                )

                # Wire up analysis
                chat_state = gr.State(value=[])

                analyze_btn.click(
                    fn=analyze_data,
                    inputs=[file_input, question_input, chat_state],
                    outputs=[chat_state, output_text, output_image, code_display],
                ).then(
                    fn=lambda h: h,
                    inputs=[chat_state],
                    outputs=[chatbot],
                )

                question_input.submit(
                    fn=analyze_data,
                    inputs=[file_input, question_input, chat_state],
                    outputs=[chat_state, output_text, output_image, code_display],
                ).then(
                    fn=lambda h: h,
                    inputs=[chat_state],
                    outputs=[chatbot],
                )

            # --- About Tab ---
            with gr.Tab("About"):
                gr.Markdown("""
## About Hermes Agent

**Hermes Agent** is a self-improving AI agent framework by [Nous Research](https://nousresearch.com).

### Key Features
- **Self-Learning Loop**: Creates skills from experience and improves them during use
- **Model Agnostic**: Works with OpenAI, Anthropic, OpenRouter, and custom endpoints
- **Multi-Platform**: Accessible via CLI, Telegram, Discord, Slack, WhatsApp, and 14+ platforms
- **40+ Built-in Tools**: Terminal, file operations, web search, browser automation, code execution
- **26 Bundled Skills**: Data science, DevOps, research, and more

### Configuration
Set these environment variables in your Space settings:

| Variable | Description |
|----------|-------------|
| `FIREWORKS_API_KEY` | Your Fireworks AI API key |
| `FIREWORKS_MODEL` | Model name (default: accounts/fireworks/models/kimi-k2p5) |

### Links
- [GitHub Repository](https://github.com/NousResearch/hermes-agent)
- [Documentation](https://docs.hermes.nousresearch.com)
""")

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
