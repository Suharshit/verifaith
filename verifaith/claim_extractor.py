import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_claims(answer: str) -> list[dict]:
    """
    Takes a RAG-generated answer string.
    Returns a list of atomic factual claims as dicts.
    
    Returns:
        [{"claim_id": 1, "text": "..."}, ...]
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a claim decomposition engine for AI evaluation.
Break down any given text into individual atomic factual claims.

Rules:
- Each claim must contain ONE fact only
- Each claim must be self-contained and understandable alone
- Each claim must be a declarative statement
- Do NOT include opinions or vague statements
- Do NOT merge two facts into one claim

Return ONLY a valid JSON array. No explanation. No markdown.
Format: [{"claim_id": 1, "text": "..."}, {"claim_id": 2, "text": "..."}]"""
            },
            {
                "role": "user",
                "content": f"Decompose this answer into atomic claims:\n\n{answer}"
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(raw)

    if isinstance(parsed, dict):
        claims = list(parsed.values())[0]
    else:
        claims = parsed

    return claims