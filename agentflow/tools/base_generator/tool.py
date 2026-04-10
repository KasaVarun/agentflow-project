import os
from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

TOOL_NAME = "Generalist_Solution_Generator_Tool"

LIMITATION = f"""
The {TOOL_NAME} may provide hallucinated or incorrect responses.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Use it for general queries or tasks that don't require specialized tools.
2. Provide clear, specific queries.
3. Use it for step-by-step reasoning on straightforward tasks.
4. Verify important information from its responses.
"""


class Base_Generator_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="together-Qwen/Qwen2.5-7B-Instruct-Turbo"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A generalized tool that takes a query and answers it step by step.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The query from the user.",
            },
            output_type="str - The generated response to the original query",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Summarize the following text in a few lines")',
                    "description": "Generate a short summary given the query from the user."
                },
            ],
            user_metadata={
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }
        )
        self.model_string = model_string
        self.llm_engine = create_llm_engine(
            model_string=self.model_string,
            is_multimodal=False,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

    def execute(self, query, image=None):
        try:
            response = self.llm_engine(query)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata
