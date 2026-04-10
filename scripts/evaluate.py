"""
Evaluate AgentFlow results using LLM-as-judge (Together AI).

Usage:
    python scripts/evaluate.py \
        --benchmark bamboogle \
        --result_dir results/bamboogle/together_Qwen_Qwen2.5_7B_Instruct \
        --response_type direct_output
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agentflow.judge import create_judge_client, judge_answer
from tqdm import tqdm


BENCHMARK_DATA_FILES = {
    "bamboogle": "benchmarks/bamboogle/data/data.json",
    "twowiki": "benchmarks/twowiki/data/data.json",
    "hotpotqa": "benchmarks/hotpotqa/data/data.json",
    "musique": "benchmarks/musique/data/data.json",
    "gaia": "benchmarks/gaia/data/data.json",
    "text2sql": "benchmarks/text2sql/data/data.json",
}


def load_data_and_results(benchmark, result_dir, response_type):
    """Load benchmark data and result files, returning matched pairs."""
    data_file = BENCHMARK_DATA_FILES[benchmark]
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Index by pid
    benchmark_data = {}
    for item in raw_data:
        pid = str(item.get("pid", item.get("idx", "")))
        benchmark_data[pid] = item

    # Load results
    if not os.path.isdir(result_dir):
        print(f"Warning: result_dir not found: {result_dir}")
        return {}
    results = {}
    for fname in os.listdir(result_dir):
        if fname.endswith(".json") and fname.startswith("output_"):
            fpath = os.path.join(result_dir, fname)
            with open(fpath, 'r') as f:
                result = json.load(f)

            index = fname.replace(".json", "").replace("output_", "")
            pid = str(result.get("pid", index))

            if pid not in benchmark_data:
                print(f"Warning: pid {pid} not in benchmark data, skipping")
                continue

            question = benchmark_data[pid].get("query", benchmark_data[pid].get("question", ""))
            gold_answer = benchmark_data[pid].get("answer", "")
            response = result.get(response_type, "")

            if not response:
                print(f"Warning: no {response_type} in output_{index}.json")
                continue

            results[pid] = {
                "question": question,
                "gold_answer": gold_answer,
                "response": response,
            }

    return results


def score_results(results, max_workers=4):
    """Score all results using LLM-as-judge."""
    client = create_judge_client()
    correct = 0
    total = len(results)
    scored = {}

    def score_one(pid_data):
        pid, data = pid_data
        score = judge_answer(
            predicted_answer=data["response"],
            gold_answer=data["gold_answer"],
            question=data["question"],
            client=client,
        )
        return pid, score

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_one, item) for item in results.items()]
        for future in tqdm(as_completed(futures), total=total, desc="Scoring"):
            pid, score = future.result()
            correct += score
            scored[pid] = {**results[pid], "score": score}

    accuracy = round(correct / total * 100, 2) if total > 0 else 0
    return scored, correct, total, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARK_DATA_FILES.keys()))
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--response_type", default="direct_output",
                        choices=["direct_output", "final_output", "base_response"])
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    print(f"Evaluating {args.benchmark} results from {args.result_dir}")

    results = load_data_and_results(args.benchmark, args.result_dir, args.response_type)
    print(f"Loaded {len(results)} results")
    if not results:
        print("No results to evaluate. Skipping.")
        return

    scored, correct, total, accuracy = score_results(results, args.max_workers)

    print(f"\n{'='*50}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Accuracy: {accuracy}% ({correct}/{total})")
    print(f"{'='*50}")

    # Save scores
    score_file = os.path.join(args.result_dir, f"scores_{args.response_type}.json")
    with open(score_file, 'w') as f:
        json.dump({
            "benchmark": args.benchmark,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": scored,
        }, f, indent=2)
    print(f"Scores saved to: {score_file}")

    # Print wrong answers
    wrong = [pid for pid, data in scored.items() if data["score"] == 0]
    if wrong:
        print(f"\nWrong answers ({len(wrong)}): {wrong[:20]}")


if __name__ == "__main__":
    main()
