"""
AgentFlow Gradio Demo - Clean version without LoRA local option
Runs AgentFlow on a user query using Together AI.

Usage:
    pip install gradio
    python demo/app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from agentflow.solver import construct_solver
from agentflow.models.memory import Memory

# ── Configuration ──────────────────────────────────────────
DEFAULT_ENGINE = "together-Qwen/Qwen2.5-7B-Instruct-Turbo"
# ───────────────────────────────────────────────────────────

_solver_cache = {}

def get_solver(engine: str, tools: list):
    key = (engine, tuple(sorted(tools)))
    if key not in _solver_cache:
        _solver_cache[key] = construct_solver(
            planner_engine=engine,
            fixed_engine=DEFAULT_ENGINE,
            enabled_tools=tools,
            output_types="direct",
            max_steps=8,
            max_time=120,
            verbose=True,
            temperature=0.0,
        )
    return _solver_cache[key]


def extract_steps(solver, result):
    """Try multiple ways to extract reasoning steps from the solver."""
    steps_lines = []

    # Method 1: result has a 'steps' key
    steps = result.get("steps", [])
    if steps:
        for i, s in enumerate(steps):
            tool = s.get("tool", s.get("action", "?"))
            content = str(s.get("result", s.get("output", s.get("content", s))))[:500]
            steps_lines.append(f"Step {i+1} [{tool}]\n{content}")
        return "\n\n---\n\n".join(steps_lines)

    # Method 2: result has 'trajectory' key
    trajectory = result.get("trajectory", [])
    if trajectory:
        for i, t in enumerate(trajectory):
            steps_lines.append(f"Step {i+1}\n{str(t)[:500]}")
        return "\n\n---\n\n".join(steps_lines)

    # Method 3: pull from solver.memory messages/history
    memory = getattr(solver, 'memory', None)
    if memory:
        messages = (
            getattr(memory, 'messages', None) or
            getattr(memory, 'history', None) or
            getattr(memory, 'turns', None) or
            []
        )
        if messages:
            for i, msg in enumerate(messages):
                if isinstance(msg, dict):
                    role = msg.get('role', '?')
                    content = str(msg.get('content', ''))[:500]
                else:
                    role = "step"
                    content = str(msg)[:500]
                steps_lines.append(f"Step {i+1} [{role}]\n{content}")
            return "\n\n---\n\n".join(steps_lines)

    # Method 4: show raw result keys as debug
    debug_info = []
    for k, v in result.items():
        if k not in ("direct_output", "final_output"):
            debug_info.append(f"[{k}]: {str(v)[:300]}")
    if debug_info:
        return "Raw result fields:\n\n" + "\n\n".join(debug_info)

    return "No intermediate steps recorded."


def run_query(query: str, engine: str, use_google: bool, use_wikipedia: bool, use_sql: bool):
    if not query.strip():
        return "Please enter a query.", ""

    tools = ["Base_Generator_Tool"]
    if use_google:
        tools.append("Google_Search_Tool")
    if use_wikipedia:
        tools.append("Wikipedia_Search_Tool")
    if use_sql:
        tools.append("SQL_Executor_Tool")

    try:
        solver = get_solver(engine, tools)
        solver.memory = Memory()
        result = solver.solve(query)

        answer = result.get("direct_output", result.get("final_output", "No answer found."))
        steps_text = extract_steps(solver, result)

        return answer, steps_text

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Error: {e}", f"Traceback:\n{tb}"


with gr.Blocks(title="AgentFlow Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# AgentFlow Demo
Multi-agent question answering with a trainable Planner + tool-using Executor.
Built for Northeastern Self-Improving AI Systems (Spring 2026).
""")

    with gr.Row():
        with gr.Column(scale=2):
            query_box = gr.Textbox(
                label="Query",
                placeholder="e.g. Who directed the movie that won the Oscar for Best Picture in 2023?",
                lines=3,
            )
            engine_dropdown = gr.Dropdown(
                label="Planner Model",
                choices=[
                    "together-Qwen/Qwen2.5-7B-Instruct-Turbo",
                    "together-meta-llama/Llama-3.3-70B-Instruct-Turbo",
                ],
                value="together-Qwen/Qwen2.5-7B-Instruct-Turbo",
            )
            with gr.Row():
                use_google = gr.Checkbox(label="Google Search", value=True)
                use_wiki = gr.Checkbox(label="Wikipedia", value=True)
                use_sql = gr.Checkbox(label="SQL Executor", value=False)
            run_btn = gr.Button("Run AgentFlow", variant="primary", size="lg")

        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Final Answer", lines=6)
            steps_box = gr.Textbox(label="Reasoning Steps", lines=14)

    gr.Examples(
        examples=[
            ["What is the capital of the country that hosted the 2022 FIFA World Cup?", "together-Qwen/Qwen2.5-7B-Instruct-Turbo", True, True, False],
            ["Which director has won more Oscars: Steven Spielberg or Martin Scorsese?", "together-Qwen/Qwen2.5-7B-Instruct-Turbo", True, True, False],
            ["How many singers are in the concert_singer database?", "together-Qwen/Qwen2.5-7B-Instruct-Turbo", False, False, True],
            ["Who was the US president when the Berlin Wall fell?", "together-Qwen/Qwen2.5-7B-Instruct-Turbo", True, True, False],
        ],
        inputs=[query_box, engine_dropdown, use_google, use_wiki, use_sql],
    )

    run_btn.click(
        fn=run_query,
        inputs=[query_box, engine_dropdown, use_google, use_wiki, use_sql],
        outputs=[answer_box, steps_box],
    )

if __name__ == "__main__":
    demo.launch(share=False, show_error=True)