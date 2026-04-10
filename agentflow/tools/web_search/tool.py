import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

load_dotenv()

TOOL_NAME = "Web_RAG_Search_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1) Requires valid URLs that are accessible and contain text content.
2) May not work with JavaScript-heavy websites or those requiring authentication.
3) Truncates long pages to fit within LLM context.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1) Use specific, targeted queries rather than broad questions.
2) Ensure the URL is accessible and contains relevant information.
3) Use it after calling other search tools to get real URLs.
"""

SUMMARIZE_PROMPT_TEMPLATE = """You are an expert AI assistant. Answer the user's query based exclusively on the provided reference text.

## User Query
{query}

## Reference Text (from {url})
{reference_text}

## Instructions
1. Identify all parts of the reference that are relevant to the query.
2. Synthesize the relevant information into a clear, concise answer.
3. If the reference does not contain relevant information, say so.

Answer:
"""


class Web_Search_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="together-Qwen/Qwen2.5-7B-Instruct-Turbo"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A tool for answering questions by retrieving and summarizing information from a given website URL.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query for the website.",
                "url": "str - The URL of the website to retrieve information from.",
            },
            output_type="str - The answer to the user's query based on website content.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the mass of the moon?", url="https://en.wikipedia.org/wiki/Moon")',
                    "description": "Retrieve information about the moon's mass from Wikipedia."
                },
            ],
            user_metadata={
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }
        )
        self.model_string = model_string
        self.max_content_chars = 12000
        self.llm_engine = create_llm_engine(
            model_string=self.model_string,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

    def _get_website_content(self, url):
        """Extract text content from a URL."""
        url = url.replace("arxiv.org/pdf", "arxiv.org/abs")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
        }
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text = soup.get_text(separator='\n', strip=True)
            return text[:self.max_content_chars]
        except Exception as e:
            return f"Error fetching URL: {str(e)}"

    def execute(self, query, url):
        """
        Fetch a webpage and use an LLM to answer the query based on its content.

        Parameters:
            query (str): The search query.
            url (str): The URL to fetch content from.

        Returns:
            str: The answer based on webpage content.
        """
        try:
            content = self._get_website_content(url)
            if content.startswith("Error"):
                return content

            prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
                query=query,
                url=url,
                reference_text=content
            )
            summary = self.llm_engine(prompt)
            return summary
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata
