"""
AgentFlow Solver - main orchestration module.
Adapted from the original AgentFlow repo to use:
  - Planner: vLLM on Modal (or Together AI for Qwen-2.5-7B reproduction)
  - Fixed engine (executor, verifier, generator): Together AI
"""
import argparse
import time
import json
import os
from typing import Optional, List, Dict, Any

from agentflow.models.initializer import Initializer
from agentflow.models.planner import Planner
from agentflow.models.verifier import Verifier
from agentflow.models.memory import Memory
from agentflow.models.executor import Executor
from agentflow.models.utils import make_json_serializable_truncated


class Solver:
    def __init__(
        self,
        planner,
        verifier,
        memory,
        executor,
        output_types: str = "base,final,direct",
        max_steps: int = 10,
        max_time: int = 300,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
        verbose: bool = True,
        temperature: float = 0.0,
    ):
        self.planner = planner
        self.verifier = verifier
        self.memory = memory
        self.executor = executor
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir
        self.output_types = output_types.lower().split(',')
        self.temperature = temperature
        self.verbose = verbose

    def solve(self, question: str, image_path: Optional[str] = None):
        """Solve a single question using the AgentFlow pipeline."""
        self.executor.set_query_cache_dir(self.root_cache_dir)

        json_data = {"query": question, "image": image_path}
        if self.verbose:
            print(f"\n==> Query: {question}")

        # Base response
        if 'base' in self.output_types:
            base_response = self.planner.generate_base_response(question, image_path, self.max_tokens)
            json_data["base_response"] = base_response
            if self.verbose:
                print(f"\n==> Base Response:\n{base_response}")

        if set(self.output_types) == {'base'}:
            return json_data

        # Full pipeline
        if {'final', 'direct'} & set(self.output_types):
            if self.verbose:
                print(f"\n==> AgentFlow Reasoning Steps:")

            # Step 1: Analyze query
            query_start_time = time.time()
            query_analysis = self.planner.analyze_query(question, image_path)
            json_data["query_analysis"] = query_analysis
            if self.verbose:
                print(f"\n==> Step 0: Query Analysis\n{query_analysis}")
                print(f"[Time]: {round(time.time() - query_start_time, 2)}s")

            # Main execution loop
            step_count = 0
            while step_count < self.max_steps and (time.time() - query_start_time) < self.max_time:
                step_count += 1
                step_start_time = time.time()

                # Step 2: Generate next step (PLANNER - trainable)
                next_step = self.planner.generate_next_step(
                    question, image_path, query_analysis,
                    self.memory, step_count, self.max_steps, json_data
                )
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
                if self.verbose:
                    print(f"\n==> Step {step_count}: Action ({tool_name})")
                    print(f"[Context]: {context}\n[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")
                    print(f"[Time]: {round(time.time() - step_start_time, 2)}s")

                if tool_name is None or tool_name not in self.planner.available_tools:
                    print(f"\n==> Error: Tool '{tool_name}' not available.")
                    command = "No command generated - tool not found."
                    result = "No result - tool not found."
                else:
                    # Step 3: Generate tool command (EXECUTOR - fixed engine)
                    tool_command = self.executor.generate_tool_command(
                        question, image_path, context, sub_goal,
                        tool_name, self.planner.toolbox_metadata[tool_name],
                        step_count, json_data
                    )
                    analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
                    if self.verbose:
                        print(f"\n==> Step {step_count}: Command ({tool_name})")
                        print(f"[Command]: {command}")

                    # Step 4: Execute tool
                    result = self.executor.execute_tool_command(tool_name, command)
                    result = make_json_serializable_truncated(result)
                    json_data[f"tool_result_{step_count}"] = result
                    if self.verbose:
                        print(f"\n==> Step {step_count}: Result")
                        print(f"[Result]: {json.dumps(result, indent=2)[:500]}")

                # Update memory
                self.memory.add_action(step_count, tool_name, sub_goal, command, result)

                # Step 5: Verify (VERIFIER - fixed engine)
                stop_verification = self.verifier.verificate_context(
                    question, image_path, query_analysis,
                    self.memory, step_count, json_data
                )
                context_verification, conclusion = self.verifier.extract_conclusion(stop_verification)
                if self.verbose:
                    print(f"\n==> Step {step_count}: Verification -> {conclusion}")

                if conclusion == 'STOP':
                    break

            # Save stats
            json_data.update({
                "memory": self.memory.get_actions(),
                "step_count": step_count,
                "execution_time": round(time.time() - query_start_time, 2),
            })

            # Generate outputs
            if 'final' in self.output_types:
                final_output = self.planner.generate_final_output(question, image_path, self.memory)
                json_data["final_output"] = final_output
                if self.verbose:
                    print(f"\n==> Final Output:\n{final_output}")

            if 'direct' in self.output_types:
                direct_output = self.planner.generate_direct_output(question, image_path, self.memory)
                json_data["direct_output"] = direct_output
                if self.verbose:
                    print(f"\n==> Direct Answer:\n{direct_output}")

            print(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")

        return json_data


def construct_solver(
    planner_engine: str = "vllm-Qwen/Qwen2.5-7B-Instruct",
    fixed_engine: str = "together-Qwen/Qwen2.5-7B-Instruct-Turbo",
    enabled_tools: List[str] = None,
    tool_engine: List[str] = None,
    output_types: str = "direct",
    max_steps: int = 10,
    max_time: int = 300,
    max_tokens: int = 4000,
    root_cache_dir: str = "solver_cache",
    verbose: bool = True,
    base_url: str = None,
    temperature: float = 0.0,
):
    """
    Construct an AgentFlow solver with our infrastructure:
      - planner_engine: Model for the planner (e.g. "vllm-Qwen/Qwen2.5-7B-Instruct")
      - fixed_engine: Model for executor/verifier/generator (e.g. "together-Qwen/Qwen2.5-7B-Instruct")
    """
    if enabled_tools is None:
        enabled_tools = [
            "Base_Generator_Tool",
            "Python_Coder_Tool",
            "Google_Search_Tool",
            "Wikipedia_Search_Tool",
        ]
    if tool_engine is None:
        tool_engine = [
            "together-Qwen/Qwen2.5-7B-Instruct-Turbo",
            "together-Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Default",
            "Default",
        ]

    # Initialize tools
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=planner_engine,
        verbose=verbose,
        base_url=base_url,
    )

    # Planner: main engine = planner_engine (vLLM on Modal), fixed = Together AI
    planner = Planner(
        llm_engine_name=planner_engine,
        llm_engine_fixed_name=fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url,
        temperature=temperature,
    )

    # Verifier: uses fixed engine
    verifier = Verifier(
        llm_engine_name=fixed_engine,
        llm_engine_fixed_name=fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        temperature=temperature,
    )

    memory = Memory()

    # Executor: uses fixed engine
    executor = Executor(
        llm_engine_name=fixed_engine,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
        tool_instances_cache=initializer.tool_instances_cache,
    )

    solver = Solver(
        planner=planner,
        verifier=verifier,
        memory=memory,
        executor=executor,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
    )
    return solver
