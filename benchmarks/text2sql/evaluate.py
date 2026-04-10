"""
Evaluation script for Text-to-SQL benchmark.

Primary metric: Execution accuracy (compares query results, not SQL strings).
Fallback metric: Exact SQL match (normalized) when databases are unavailable.

Usage:
    python benchmarks/text2sql/evaluate.py \
        --result_dir results/text2sql/qwen35_0.8b
"""
import argparse
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SPIDER_DB_ROOT = os.getenv("SPIDER_DB_PATH", "benchmarks/text2sql/data/spider/database")
DATA_FILE = "benchmarks/text2sql/data/data.json"


def execute_sql(db_id: str, sql: str):
    """Execute SQL against Spider SQLite database. Returns sorted rows or None on error."""
    db_path = os.path.join(SPIDER_DB_ROOT, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return sorted([
            tuple(str(v).lower().strip() if v is not None else "" for v in row)
            for row in rows
        ])
    except Exception:
        return None


def normalize_sql(sql: str) -> str:
    """Normalize SQL for string comparison."""
    sql = re.sub(r"\s+", " ", sql.strip().lower())
    sql = sql.rstrip(";").strip()
    # Remove table aliases: "FROM singer AS T1" -> "FROM singer"
    sql = re.sub(r"\bas\s+t\d+\b", "", sql)
    return sql


def extract_sql(response: str) -> str:
    """Extract SQL from <answer> tags, code blocks, or raw SELECT."""
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Find first SELECT statement
    lines = response.strip().split("\n")
    sql_lines = []
    in_sql = False
    for line in lines:
        if line.strip().upper().startswith("SELECT"):
            in_sql = True
        if in_sql:
            sql_lines.append(line)
            if line.strip().endswith(";"):
                break
    if sql_lines:
        return " ".join(sql_lines).strip().rstrip(";")

    return response.strip()


def evaluate_results(result_dir: str):
    """Evaluate results. Uses execution accuracy if DBs present, else exact match."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    benchmark = {str(item["pid"]): item for item in raw_data}

    # Check if databases are available
    has_dbs = os.path.isdir(SPIDER_DB_ROOT) and len(os.listdir(SPIDER_DB_ROOT)) > 5
    metric = "execution_accuracy" if has_dbs else "exact_match_normalized"
    print(f"Evaluation metric: {metric}")
    if not has_dbs:
        print(f"[warn] No databases at {SPIDER_DB_ROOT}. Using exact-match fallback.")
        print("[warn] Run: python benchmarks/text2sql/download_spider.py")

    exec_correct = 0
    match_correct = 0
    total = 0
    details = {}
    errors = []

    for fname in sorted(os.listdir(result_dir)):
        if not (fname.startswith("output_") and fname.endswith(".json")):
            continue

        fpath = os.path.join(result_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            result = json.load(f)

        index = fname.replace("output_", "").replace(".json", "")
        pid = str(result.get("pid", index))
        if pid not in benchmark:
            continue

        item = benchmark[pid]
        db_id = item.get("db_id", "")
        gold_sql = item.get("gold_sql", "")

        predicted_response = result.get("direct_output", result.get("final_output", ""))
        predicted_sql = extract_sql(predicted_response)

        total += 1

        # Execution accuracy (preferred)
        exec_match = False
        if has_dbs:
            gold_rows = execute_sql(db_id, gold_sql)
            pred_rows = execute_sql(db_id, predicted_sql)
            if gold_rows is not None and pred_rows is not None:
                exec_match = (gold_rows == pred_rows)
            elif gold_rows is None:
                total -= 1  # skip unevaluable
                continue
            if exec_match:
                exec_correct += 1

        # Exact match (always computed for reference)
        exact_match = normalize_sql(predicted_sql) == normalize_sql(gold_sql)
        if exact_match:
            match_correct += 1

        if not (exec_match if has_dbs else exact_match):
            errors.append({
                "pid": pid,
                "question": item.get("question", ""),
                "gold_sql": gold_sql,
                "predicted_sql": predicted_sql,
            })

        details[pid] = {
            "question": item.get("question", ""),
            "db_id": db_id,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "exec_match": exec_match if has_dbs else None,
            "exact_match": exact_match,
        }

    if total == 0:
        print("No results found.")
        return 0.0

    exec_acc = round(exec_correct / total * 100, 2) if has_dbs else None
    match_acc = round(match_correct / total * 100, 2)
    primary_acc = exec_acc if has_dbs else match_acc

    print(f"{'='*50}")
    print(f"Text-to-SQL Evaluation")
    print(f"Result dir: {result_dir}")
    if has_dbs:
        print(f"Execution Accuracy: {exec_acc}% ({exec_correct}/{total})")
    print(f"Exact Match (normalized): {match_acc}% ({match_correct}/{total})")
    print(f"{'='*50}")

    score_file = os.path.join(result_dir, "scores_text2sql.json")
    with open(score_file, "w") as f:
        json.dump({
            "benchmark": "text2sql_spider",
            "metric": metric,
            "execution_accuracy": exec_acc,
            "exact_match_accuracy": match_acc,
            "accuracy": primary_acc,
            "correct": exec_correct if has_dbs else match_correct,
            "total": total,
            "details": details,
            "errors_sample": errors[:10],
        }, f, indent=2)
    print(f"Scores saved to: {score_file}")
    return primary_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    args = parser.parse_args()
    evaluate_results(args.result_dir)


if __name__ == "__main__":
    main()
