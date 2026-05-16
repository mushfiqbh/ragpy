from __future__ import annotations
import os
import requests
from typing import List

def embed(text: str) -> List[float]:
    cleaned = (text or "").strip()
    # text-embedding-3-small dimension is 1536
    if not cleaned:
        return [0.0] * 1536  

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not set. Returning zeros.")
        return [0.0] * 1536

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/text-embedding-3-small", 
        "input": cleaned
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"OpenRouter embedding failed: {e}. Returning zeros.")
        return [0.0] * 1536
