"""
LLM-as-Judge for AgentFlow evaluation.
Uses Together AI (Qwen-2.5-7B-Instruct) for binary correctness judgment.
"""
import os
import re
from openai import OpenAI


def create_judge_client():
    """Create an OpenAI-compatible client pointing to Together AI."""
    return OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )


def judge_answer(predicted_answer: str, gold_answer, question: str, client=None) -> int:
    """
    Binary 0/1 judgment using Together AI Qwen-2.5-7B-Instruct.

    Args:
        predicted_answer: The model's predicted answer.
        gold_answer: The correct answer (str or list of acceptable answers).
        question: The original question.
        client: Optional pre-created OpenAI client.

    Returns:
        1 if correct, 0 if incorrect.
    """
    if client is None:
        client = create_judge_client()

    # Handle list of acceptable answers
    if isinstance(gold_answer, list):
        gold_str = " OR ".join(str(a) for a in gold_answer)
    else:
        gold_str = str(gold_answer)

    # Extract answer from <answer> tags if present
    all_matches = re.findall(r"<answer>(.*?)</answer>", str(predicted_answer), re.DOTALL)
    if all_matches:
        predicted_answer = all_matches[-1].strip()

    prompt = f"""You are an expert evaluator. Given a question and two answers, determine if the predicted answer is correct.

Question: {question}
Gold Answer: {gold_str}
Predicted Answer: {predicted_answer}

Instructions:
- The predicted answer is correct if it conveys the same meaning as the gold answer.
- Allow minor formatting differences (capitalization, punctuation, extra words).
- The predicted answer does NOT need to be an exact string match.
- Focus on whether the core factual content matches.

Is the predicted answer correct? Respond with exactly "1" if correct, "0" if incorrect. No other text."""

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
        return 1 if "1" in result else 0
    except Exception as e:
        print(f"Judge error: {e}")
        return 0
