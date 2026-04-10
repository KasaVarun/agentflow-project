"""
SQL Executor Tool for AgentFlow Text-to-SQL benchmark.
Executes SQL queries against the Spider SQLite databases and returns results.
"""
import os
import sqlite3
import json
from agentflow.tools.base import BaseTool

TOOL_NAME = "SQL_Executor_Tool"

LIMITATION = f"""
{TOOL_NAME} has the following limitations:
1. Only supports SQLite (Spider benchmark databases).
2. Only executes SELECT queries (no modification queries for safety).
3. Requires the Spider database files to be present locally.
4. Results are limited to 50 rows.
"""

BEST_PRACTICE = f"""
For optimal results with {TOOL_NAME}:
1. First understand the database schema (table names, column names, types).
2. Write the SQL query to answer the natural language question.
3. Use the exact table and column names from the schema.
4. Always use SELECT queries, not INSERT/UPDATE/DELETE.
5. The tool returns execution results as a list of rows.
"""


class SQL_Executor_Tool(BaseTool):
    # Spider database root - set via SPIDER_DB_PATH env var
    DB_ROOT = os.getenv("SPIDER_DB_PATH", "benchmarks/text2sql/data/spider/database")

    def __init__(self, model_string=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Executes a SQL query against a SQLite database from the Spider benchmark. "
                "Returns the query results as a list of rows."
            ),
            tool_version="1.0.0",
            input_types={
                "db_id": "str - The Spider database ID (e.g., 'concert_singer').",
                "sql": "str - The SQL query to execute.",
            },
            output_type="str - Query results as JSON list of rows, or error message.",
            demo_commands=[
                {
                    "command": (
                        'execution = tool.execute(db_id="concert_singer", '
                        'sql="SELECT COUNT(*) FROM singer")'
                    ),
                    "description": "Count singers in the concert_singer database.",
                },
            ],
            user_metadata={
                "limitations": LIMITATION,
                "best_practices": BEST_PRACTICE,
            }
        )

    def _get_db_path(self, db_id: str) -> str:
        """Resolve the SQLite file path for a given db_id."""
        db_file = os.path.join(self.DB_ROOT, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_file):
            # Try flat layout
            db_file_flat = os.path.join(self.DB_ROOT, f"{db_id}.sqlite")
            if os.path.exists(db_file_flat):
                return db_file_flat
            raise FileNotFoundError(
                f"Database not found: {db_file}\n"
                f"Set SPIDER_DB_PATH or download Spider dataset to "
                f"benchmarks/text2sql/data/spider/"
            )
        return db_file

    def get_schema(self, db_id: str) -> str:
        """Return CREATE TABLE statements for a database (useful for context)."""
        db_path = self._get_db_path(db_id)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        schema = "\n\n".join(row[0] for row in rows if row[0])
        return schema

    def execute(self, db_id: str, sql: str) -> str:
        """
        Execute a SQL query against the Spider database.

        Args:
            db_id: Spider database name (e.g., 'concert_singer')
            sql: SQL SELECT query

        Returns:
            JSON string of results or error message
        """
        # Safety: only allow SELECT
        sql_stripped = sql.strip().lower()
        if not sql_stripped.startswith("select"):
            return json.dumps({"error": "Only SELECT queries are allowed."})

        try:
            db_path = self._get_db_path(db_id)
        except FileNotFoundError as e:
            return json.dumps({"error": str(e)})

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(50)  # limit results
            conn.close()

            results = [dict(row) for row in rows]
            return json.dumps(results, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": f"SQL execution error: {str(e)}", "sql": sql})


if __name__ == "__main__":
    tool = SQL_Executor_Tool()
    print(tool.execute(db_id="concert_singer", sql="SELECT COUNT(*) FROM singer"))
