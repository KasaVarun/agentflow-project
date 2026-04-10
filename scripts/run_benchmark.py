"""
Benchmark runner for AgentFlow.
Runs the AgentFlow solver on a benchmark dataset and saves results.

Usage:
    python scripts/run_benchmark.py \
        --benchmark bamboogle \
        --planner_engine "together-Qwen/Qwen2.5-7B-Instruct" \
        --index 0

    # Run all indices:
    python scripts/run_benchmark.py \
        --benchmark bamboogle \
        --planner_engine "together-Qwen/Qwen2.5-7B-Instruct" \
        --run_all

    # With Modal planner:
    python scripts/run_benchmark.py \
        --benchmark bamboogle \
        --planner_engine "vllm-Qwen/Qwen2.5-7B-Instruct" \
        --planner_base_url "https://your-modal-app--serve.modal.run/v1"
"""
import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agentflow.solver import construct_solver
from agentflow.models.memory import Memory


BENCHMARK_CONFIG = {
    "bamboogle": {
        "data_file": "benchmarks/bamboogle/data/data.json",
        "description": "Multi-hop factual QA (125 questions)",
    },
    "twowiki": {
        "data_file": "benchmarks/twowiki/data/data.json",
        "description": "2WikiMultiHopQA subset",
    },
    "hotpotqa": {
        "data_file": "benchmarks/hotpotqa/data/data.json",
        "description": "HotpotQA multi-hop reasoning",
    },
    "musique": {
        "data_file": "benchmarks/musique/data/data.json",
        "description": "Musique multi-hop QA",
    },
    "gaia": {
        "data_file": "benchmarks/gaia/data/data.json",
        "description": "GAIA general AI assistant benchmark",
    },
    "text2sql": {
        "data_file": "benchmarks/text2sql/data/data.json",
        "description": "Spider Text-to-SQL (execution accuracy)",
    },
}


def load_benchmark_data(benchmark_name: str) -> list:
    config = BENCHMARK_CONFIG[benchmark_name]
    data_file = config["data_file"]
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Normalize field names
    for item in data:
        if 'query' not in item and 'question' in item:
            item['query'] = item['question']
    return data


def run_single(solver, data, index, output_dir):
    """Run solver on a single problem and save output."""
    if index >= len(data):
        print(f"Index {index} out of range (max {len(data) - 1})")
        return None

    problem = data[index]
    question = problem.get("query", problem.get("question", ""))
    image_path = problem.get("image", None)
    pid = problem.get("pid", str(index))
    answer = problem.get("answer", "")

    print(f"\n{'='*80}")
    print(f"Problem {index} (pid={pid})")
    print(f"Q: {question[:200]}")
    print(f"Gold: {answer}")
    print(f"{'='*80}")

    # Reset memory for each problem
    solver.memory = Memory()

    result = solver.solve(question, image_path)
    result["pid"] = pid
    result["answer"] = answer

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_{index}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run AgentFlow on a benchmark")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARK_CONFIG.keys()))
    parser.add_argument("--planner_engine", default="together-Qwen/Qwen2.5-7B-Instruct-Turbo",
                        help="Planner engine string (e.g. vllm-Qwen/Qwen2.5-7B-Instruct-Turbo)")
    parser.add_argument("--fixed_engine", default="together-Qwen/Qwen2.5-7B-Instruct-Turbo",
                        help="Fixed engine for executor/verifier/generator")
    parser.add_argument("--planner_base_url", default=None,
                        help="Base URL for vLLM planner on Modal")
    parser.add_argument("--index", type=int, default=None, help="Single problem index")
    parser.add_argument("--start", type=int, default=0, help="Start index for range")
    parser.add_argument("--end", type=int, default=None, help="End index for range")
    parser.add_argument("--run_all", action="store_true", help="Run all problems")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--exp_name", default=None, help="Experiment name for output dir")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_time", type=int, default=300)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip problems that already have output files")
    parser.add_argument("--tools", default=None,
                        help="Comma-separated tools to enable. Default: all four. "
                             "E.g. --tools Wikipedia_Search_Tool,Base_Generator_Tool")
    parser.add_argument("--no_google", action="store_true",
                        help="Exclude Google_Search_Tool (use if CSE API is unavailable)")
    parser.add_argument("--sql", action="store_true",
                        help="Include SQL_Executor_Tool (for text2sql benchmark)")
    args = parser.parse_args()

    # Set planner base URL in environment if provided
    if args.planner_base_url:
        os.environ["PLANNER_BASE_URL"] = args.planner_base_url

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        exp_name = args.exp_name or args.planner_engine.replace("/", "_").replace("-", "_")
        output_dir = os.path.join("results", args.benchmark, exp_name)

    # Determine tools
    if args.tools:
        enabled_tools = [t.strip() for t in args.tools.split(",")]
    elif args.benchmark == "text2sql":
        # SQL benchmark: use SQL executor + base generator, no search tools needed
        enabled_tools = [
            "Base_Generator_Tool",
            "SQL_Executor_Tool",
        ]
    elif args.no_google:
        enabled_tools = [
            "Base_Generator_Tool",
            "Python_Coder_Tool",
            "Wikipedia_Search_Tool",
        ]
    else:
        enabled_tools = [
            "Base_Generator_Tool",
            "Python_Coder_Tool",
            "Google_Search_Tool",
            "Wikipedia_Search_Tool",
        ]
    # Optionally add SQL tool to any benchmark
    if args.sql and "SQL_Executor_Tool" not in enabled_tools:
        enabled_tools.append("SQL_Executor_Tool")

    print(f"Benchmark: {args.benchmark}")
    print(f"Planner: {args.planner_engine}")
    print(f"Fixed engine: {args.fixed_engine}")
    print(f"Tools: {enabled_tools}")
    print(f"Output: {output_dir}")
    if args.planner_base_url:
        print(f"Planner URL: {args.planner_base_url}")

    # Load data
    data = load_benchmark_data(args.benchmark)
    print(f"Loaded {len(data)} problems")

    # Construct solver
    solver = construct_solver(
        planner_engine=args.planner_engine,
        fixed_engine=args.fixed_engine,
        enabled_tools=enabled_tools,
        output_types="direct",
        max_steps=args.max_steps,
        max_time=args.max_time,
        verbose=args.verbose,
        base_url=args.planner_base_url,
        temperature=0.0,
    )

    # Determine which indices to run
    if args.index is not None:
        indices = [args.index]
    elif args.run_all:
        indices = list(range(len(data)))
    else:
        end = args.end if args.end is not None else len(data)
        indices = list(range(args.start, min(end, len(data))))

    # Skip existing
    if args.skip_existing:
        new_indices = []
        for i in indices:
            output_file = os.path.join(output_dir, f"output_{i}.json")
            if not os.path.exists(output_file):
                new_indices.append(i)
            else:
                print(f"Skipping index {i} (already exists)")
        indices = new_indices

    print(f"Running {len(indices)} problems: {indices[:10]}{'...' if len(indices) > 10 else ''}")

    # Run
    start_time = time.time()
    completed = 0
    for i in indices:
        try:
            run_single(solver, data, i, output_dir)
            completed += 1
        except Exception as e:
            print(f"Error on index {i}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nCompleted {completed}/{len(indices)} problems in {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
