import ollama
import json
import re
from config import OLLAMA_MODEL, SYSTEM_PROMPT


def get_llm_recommendations(user_query: str) -> list:
    """
    Send user query to Ollama and get LLM recommendations as a list of dicts.
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
        )

        raw_text = response["message"]["content"]
        recommendations = parse_json_response(raw_text)
        return recommendations

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return []


def parse_json_response(raw_text: str) -> list:
    """
    Extract and parse JSON array from model response.
    Handles cases where model adds extra text around the JSON.
    """
    try:
        # First try direct parse
        return json.loads(raw_text)

    except json.JSONDecodeError:
        # Try to extract JSON array using regex
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                print("Failed to parse extracted JSON.")
                return []
        print("No JSON array found in response.")
        return []