"""
Test script to verify all API connections work.
Run this before starting any experiments.

Usage:
    python scripts/test_setup.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_env_vars():
    """Check that required environment variables are set."""
    print("\n[1/5] Checking environment variables...")
    required = {
        "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID"),
    }
    all_ok = True
    for name, value in required.items():
        if value and value != "dummy" and len(value) > 5:
            print(f"  {name}: OK ({value[:8]}...)")
        else:
            print(f"  {name}: MISSING or invalid")
            all_ok = False
    return all_ok


def test_together_api():
    """Test Together AI API connectivity."""
    print("\n[2/5] Testing Together AI API...")
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=10,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        print(f"  Response: '{answer}'")
        print(f"  Together AI: OK")
        return True
    except Exception as e:
        print(f"  Together AI: FAILED - {e}")
        return False


def test_together_engine():
    """Test our ChatTogether engine wrapper."""
    print("\n[3/5] Testing ChatTogether engine wrapper...")
    try:
        from agentflow.engine.together import ChatTogether
        engine = ChatTogether(model_string="Qwen/Qwen2.5-7B-Instruct-Turbo")
        response = engine("What is the capital of France? Answer in one word.")
        if isinstance(response, dict) and "error" in response:
            print(f"  ChatTogether engine: FAILED - {response}")
            return False
        print(f"  Response: '{str(response)[:100]}'")

        # Test structured output
        from agentflow.models.formatters import MemoryVerification
        response2 = engine(
            "Is 2+2=4? Respond with analysis and stop_signal.",
            response_format=MemoryVerification
        )
        if isinstance(response2, dict) and "error" in response2:
            print(f"  Structured output: FAILED - {response2}")
            return False
        print(f"  Structured response type: {type(response2).__name__}")
        if isinstance(response2, MemoryVerification):
            print(f"  Analysis: {response2.analysis[:80]}")
            print(f"  Stop signal: {response2.stop_signal}")
        print(f"  ChatTogether engine: OK")
        return True
    except Exception as e:
        print(f"  ChatTogether engine: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_google_search():
    """Test Google Custom Search API."""
    print("\n[4/5] Testing Google Custom Search API...")
    try:
        import requests
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": "capital of France",
            "num": 1,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 403:
            data = resp.json()
            msg = data.get("error", {}).get("message", "403 Forbidden")
            print(f"  Google Search: NEEDS SETUP - {msg}")
            print(f"  ACTION REQUIRED: Enable 'Custom Search JSON API' at:")
            print(f"  https://console.cloud.google.com/apis/library/customsearch.googleapis.com")
            print(f"  Then link it to API key: {api_key[:20]}...")
            print(f"  NOTE: Benchmarks will use Wikipedia search as fallback (still works).")
            return False
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if items:
            print(f"  First result: {items[0].get('title', 'No title')}")
            print(f"  Google Search: OK")
            return True
        else:
            print(f"  Google Search: No results returned")
            return False
    except Exception as e:
        print(f"  Google Search: FAILED - {e}")
        return False


def test_judge():
    """Test the LLM-as-judge functionality."""
    print("\n[5/5] Testing LLM-as-Judge...")
    try:
        from agentflow.judge import judge_answer
        # Should be correct
        score1 = judge_answer("Paris", "Paris", "What is the capital of France?")
        # Should be incorrect
        score2 = judge_answer("London", "Paris", "What is the capital of France?")
        print(f"  Correct answer test: {score1} (expected 1)")
        print(f"  Wrong answer test: {score2} (expected 0)")
        if score1 == 1 and score2 == 0:
            print(f"  Judge: OK")
            return True
        else:
            print(f"  Judge: UNEXPECTED RESULTS")
            return False
    except Exception as e:
        print(f"  Judge: FAILED - {e}")
        return False


def main():
    print("=" * 60)
    print("AgentFlow Setup Verification")
    print("=" * 60)

    results = {}
    results["env_vars"] = test_env_vars()
    results["together_api"] = test_together_api()
    results["together_engine"] = test_together_engine()
    results["google_search"] = test_google_search()
    results["judge"] = test_judge()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed! Ready to run experiments.")
    else:
        print("\nSome tests failed. Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
