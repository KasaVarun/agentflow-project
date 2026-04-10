"""
Show all benchmark results in a summary table.

Usage:
    python scripts/show_results.py
    python scripts/show_results.py --benchmark bamboogle
"""
import argparse
import json
import os

BENCHMARKS = ["bamboogle", "twowiki", "hotpotqa", "musique", "gaia"]
TARGETS = {
    "bamboogle": 58.4,
    "twowiki": 60.0,
    "hotpotqa": 51.3,
    "musique": 19.2,
    "gaia": 17.2,
}


def find_scores(results_root="results"):
    rows = []
    for benchmark in os.listdir(results_root) if os.path.isdir(results_root) else []:
        bench_dir = os.path.join(results_root, benchmark)
        if not os.path.isdir(bench_dir):
            continue
        for exp in os.listdir(bench_dir):
            exp_dir = os.path.join(bench_dir, exp)
            score_file = os.path.join(exp_dir, "scores_direct_output.json")
            if os.path.exists(score_file):
                with open(score_file, encoding="utf-8") as f:
                    data = json.load(f)
                rows.append({
                    "benchmark": benchmark,
                    "exp": exp,
                    "accuracy": data.get("accuracy", 0),
                    "correct": data.get("correct", 0),
                    "total": data.get("total", 0),
                    "target": TARGETS.get(benchmark, "-"),
                })
            else:
                # Count raw outputs
                n_outputs = len([f for f in os.listdir(exp_dir) if f.startswith("output_") and f.endswith(".json")])
                if n_outputs > 0:
                    rows.append({
                        "benchmark": benchmark,
                        "exp": exp,
                        "accuracy": "pending",
                        "correct": "-",
                        "total": n_outputs,
                        "target": TARGETS.get(benchmark, "-"),
                    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--results_root", default="results")
    args = parser.parse_args()

    rows = find_scores(args.results_root)
    if args.benchmark:
        rows = [r for r in rows if r["benchmark"] == args.benchmark]

    if not rows:
        print("No results found.")
        return

    rows.sort(key=lambda r: (r["benchmark"], r["exp"]))

    # Print table
    print(f"{'Benchmark':<12} {'Experiment':<35} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'Target':>8}")
    print("-" * 85)
    for r in rows:
        acc = f"{r['accuracy']}%" if isinstance(r['accuracy'], float) else r['accuracy']
        print(f"{r['benchmark']:<12} {r['exp']:<35} {acc:>10} {str(r['correct']):>8} {str(r['total']):>6} {str(r['target']):>8}")


if __name__ == "__main__":
    main()
