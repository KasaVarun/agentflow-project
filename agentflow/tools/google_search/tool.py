import os
import requests
from dotenv import load_dotenv

load_dotenv()

from agentflow.tools.base import BaseTool

TOOL_NAME = "Ground_Google_Search_Tool"

LIMITATIONS = """
1. This tool is only suitable for general information search.
2. Limited to queries per Serper.dev plan limits.
3. Returns text snippets, not full page content.
"""

BEST_PRACTICES = """
1. Choose this tool when you want to search general information about a topic.
2. Choose this tool for factual queries like "What is the capital of France?"
3. The tool will return search result snippets with titles and links.
4. Use specific, well-formed queries for best results.
"""


class Google_Search_Tool(BaseTool):
    def __init__(self, model_string=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A web search tool powered by Serper.dev (Google Search results) that provides real-time information from the internet.",
            tool_version="1.1.0",
            input_types={
                "query": "str - The search query to find information on the web.",
            },
            output_type="str - The search results of the query.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the capital of France?")',
                    "description": "Search for general information about the capital of France."
                },
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            }
        )
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Serper search requires SERPER_API_KEY environment variable."
            )

    def _search(self, query: str, num_results: int = 5) -> list:
        """Call Serper.dev Google Search API."""
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": min(num_results, 10)}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # Serper returns organic results under "organic" key
            items = data.get("organic", [])
            # Also include answer box / knowledge graph if present
            extras = []
            if data.get("answerBox"):
                ab = data["answerBox"]
                extras.append({
                    "title": ab.get("title", "Answer Box"),
                    "snippet": ab.get("answer") or ab.get("snippet", ""),
                    "link": ab.get("link", ""),
                })
            return extras + items
        except Exception as e:
            return [{"title": "Error", "snippet": str(e), "link": ""}]

    def execute(self, query: str, num_results: int = 5) -> str:
        """
        Execute the Google Search tool via Serper.dev.

        Parameters:
            query (str): The search query.
            num_results (int): Number of results to return (max 10).

        Returns:
            str: Formatted search results.
        """
        items = self._search(query, num_results)
        if not items:
            return f"No results found for: {query}"

        results = []
        for i, item in enumerate(items[:num_results], 1):
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No snippet")
            link = item.get("link", "")
            results.append(f"[{i}] {title}\n{snippet}\nURL: {link}")

        return "\n\n".join(results)

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    tool = Google_Search_Tool()
    result = tool.execute(query="What is the capital of France?")
    print(result)
