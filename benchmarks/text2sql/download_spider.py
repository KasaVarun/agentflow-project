"""
Download and prepare the Spider Text-to-SQL dataset.

Spider dataset: https://yale-lily.github.io/spider
- Questions/SQL: downloaded from HuggingFace datasets
- SQLite databases + tables.json: downloaded from official Google Drive release

Usage:
    python benchmarks/text2sql/download_spider.py

This creates:
    benchmarks/text2sql/data/data.json       - eval format (1034 dev questions)
    benchmarks/text2sql/data/spider/tables.json     - schema info
    benchmarks/text2sql/data/spider/database/       - SQLite databases
"""
import json
import os
import sys
import zipfile

DATA_DIR = "benchmarks/text2sql/data"
SPIDER_DIR = os.path.join(DATA_DIR, "spider")
DB_DIR = os.path.join(SPIDER_DIR, "database")
TABLES_FILE = os.path.join(SPIDER_DIR, "tables.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "data.json")

# Official Spider release on Google Drive (includes databases + tables.json)
SPIDER_GDRIVE_ID = "1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m"


def download_databases():
    """Download Spider databases from Google Drive via gdown."""
    os.makedirs(SPIDER_DIR, exist_ok=True)
    zip_path = os.path.join(SPIDER_DIR, "spider.zip")

    if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR) and len(os.listdir(DB_DIR)) > 10:
        print(f"Databases already present at {DB_DIR}")
        return True

    print("Downloading Spider databases from Google Drive...")
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={SPIDER_GDRIVE_ID}"
        gdown.download(url, zip_path, quiet=False)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract database/ and tables.json from the zip
            members = zf.namelist()
            for member in members:
                # The zip typically has spider/database/... structure
                if "database" in member or "tables.json" in member:
                    # Flatten: spider/database/X -> database/X
                    parts = member.split("/")
                    try:
                        db_idx = parts.index("database")
                        rel_path = os.path.join(*parts[db_idx:]) if len(parts) > db_idx else None
                    except ValueError:
                        if "tables.json" in member:
                            rel_path = "tables.json"
                        else:
                            continue

                    if rel_path:
                        target = os.path.join(SPIDER_DIR, rel_path)
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())

        os.remove(zip_path)
        print(f"Databases extracted to {DB_DIR}")
        return True

    except Exception as e:
        print(f"[warn] gdown download failed: {e}")
        print("Manual download option:")
        print(f"  1. Download from: https://drive.google.com/uc?id={SPIDER_GDRIVE_ID}")
        print(f"  2. Extract database/ and tables.json to: {SPIDER_DIR}/")
        return False


def load_schema():
    """Load table schema from tables.json. Returns dict keyed by db_id."""
    if not os.path.exists(TABLES_FILE):
        return {}
    with open(TABLES_FILE, encoding="utf-8") as f:
        tables_data = json.load(f)
    schema = {}
    for db in tables_data:
        db_id = db["db_id"]
        table_names = db.get("table_names_original", db.get("table_names", []))
        col_names = db.get("column_names_original", db.get("column_names", []))
        col_types = db.get("column_types", [])

        # Build per-table column list
        tables = {t: [] for t in table_names}
        for i, (table_idx, col_name) in enumerate(col_names):
            if table_idx >= 0:  # -1 = wildcard column
                tname = table_names[table_idx]
                ctype = col_types[i] if i < len(col_types) else "text"
                tables[tname].append(f"{col_name} ({ctype})")

        # Format as DDL-like string
        ddl_parts = []
        for tname, cols in tables.items():
            cols_str = ", ".join(cols) if cols else "*"
            ddl_parts.append(f"Table {tname}: [{cols_str}]")
        schema[db_id] = " | ".join(ddl_parts)
    return schema


def build_dataset(schema: dict):
    """Build data.json from HuggingFace spider validation split."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("Loading Spider validation split from HuggingFace...")
    ds = load_dataset("spider")
    dev_data = ds["validation"]
    print(f"Validation split: {len(dev_data)} examples")

    samples = []
    for i, ex in enumerate(dev_data):
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"]

        schema_str = schema.get(db_id, "")
        schema_hint = f"\nDatabase schema: {schema_str}" if schema_str else ""

        query = (
            f"Convert this natural language question to a SQL query for the '{db_id}' database."
            f"{schema_hint}\n\n"
            f"Question: {question}\n\n"
            f"When ready, output the final SQL enclosed in <answer> and </answer> tags. "
            f"Only output SELECT statements."
        )

        samples.append({
            "idx": i,
            "pid": str(i),
            "db_id": db_id,
            "question": question,
            "query": query,
            "gold_sql": gold_sql,
            "answer": gold_sql,
            "image": None,
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {OUTPUT_FILE}")
    return samples


def main():
    # Step 1: Download databases (for execution accuracy evaluation)
    has_dbs = download_databases()
    if not has_dbs:
        print("[warn] Databases not available. Execution accuracy evaluation will be skipped.")
        print("[warn] Using LLM-as-judge fallback in evaluate.py.")

    # Step 2: Load schema (from tables.json if available)
    schema = load_schema()
    if schema:
        print(f"Loaded schema for {len(schema)} databases")
    else:
        print("[warn] No tables.json found - queries will not include schema hints.")

    # Step 3: Build data.json
    build_dataset(schema)

    # Summary
    print("\nSetup complete.")
    print(f"  Questions: {OUTPUT_FILE}")
    print(f"  Databases: {DB_DIR} ({'present' if has_dbs else 'MISSING - download manually'})")
    print(f"  Schema:    {TABLES_FILE} ({'present' if schema else 'MISSING'})")
    print()
    print("Run benchmark:")
    print("  python scripts/run_benchmark.py --benchmark text2sql --planner_engine ... --run_all")
    print()
    print("Evaluate:")
    print("  python benchmarks/text2sql/evaluate.py --result_dir results/text2sql/<exp_name>")


if __name__ == "__main__":
    main()
