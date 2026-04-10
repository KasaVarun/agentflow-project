"""
AgentFlow Gradio Demo
Runs AgentFlow on a user query using the deployed Modal endpoint or Together AI.

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
DEFAULT_TOOLS = ["Base_Generator_Tool", "Google_Search_Tool", "Wikipedia_Search_Tool"]
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
            verbose=False,
            temperature=0.0,
        )
    return _solver_cache[key]


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
        steps = result.get("steps", [])
        steps_text = "\n\n".join(
            f"**Step {i+1} [{s.get('tool', '?')}]**\n{str(s.get('result', ''))[:300]}"
            for i, s in enumerate(steps)
        ) if steps else "No intermediate steps recorded."
        return answer, steps_text
    except Exception as e:
        return f"Error: {e}", ""


with gr.Blocks(title="AgentFlow Demo") as demo:
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
            with gr.Row():
                engine_dropdown = gr.Dropdown(
                    label="Planner Model",
                    choices=[
                        "together-Qwen/Qwen2.5-7B-Instruct-Turbo",
                        "together-Qwen/Qwen3.5-9B",
                    ],
                    value="together-Qwen/Qwen2.5-7B-Instruct-Turbo",
                )
            with gr.Row():
                use_google = gr.Checkbox(label="Google Search", value=True)
                use_wiki = gr.Checkbox(label="Wikipedia", value=True)
                use_sql = gr.Checkbox(label="SQL Executor", value=False)
            run_btn = gr.Button("Run AgentFlow", variant="primary")

        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Final Answer", lines=4)
            steps_box = gr.Textbox(label="Reasoning Steps", lines=12)

    gr.Examples(
        examples=[
            ["What is the capital of the country that hosted the 2022 FIFA World Cup?", "together-Qwen/Qwen2.5-7B-Instruct-Turbo", True, True, False],
            ["Which director has won more Oscars: Steven Spielberg or Martin Scorsese?", "together-Qwen/Qwen2.5-7B-Instruct-Turbo", True, True, False],
            ["How many singers are in the concert_singer database?", "together-Qwen/Qwen2.5-7B-Instruct-Turbo", False, False, True],
        ],
        inputs=[query_box, engine_dropdown, use_google, use_wiki, use_sql],
    )

    run_btn.click(
        fn=run_query,
        inputs=[query_box, engine_dropdown, use_google, use_wiki, use_sql],
        outputs=[answer_box, steps_box],
    )

if __name__ == "__main__":
    demo.launch(share=False)
