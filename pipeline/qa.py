from __future__ import annotations

import os
import requests
from typing import Dict, List
import logging

from pipeline.embedder import embed
from pipeline.store import search

GEN_MODEL = os.getenv("RAG_GENERATION_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
USE_GENERATION = os.getenv("RAG_USE_GENERATION", "true").lower() == "true"

def _fallback_answer(query: str, docs: List[Dict]) -> str:
    if not docs:
        return "I could not find relevant context in the indexed documents."

    snippets = [d.get("parent_text") or d.get("child_text") for d in docs[:2] if d.get("parent_text") or d.get("child_text")]
    joined = "\n\n".join(f"- {s}" for s in snippets)
    return f"Based on the indexed documents, the most relevant context for '{query}' is:\n{joined}"

def answer(query: str, top_k: int = 5) -> Dict:
    query = (query or "").strip()
    if not query:
        return {"answer": "Question is empty.", "contexts": []}

    query_vec = embed(query)
    # Fetch top matches
    docs = search(query, query_vec, top_k=top_k)
    docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    best_docs = docs[:top_k]
    
    context = "\n\n".join([d.get("parent_text") or d.get("child_text") for d in best_docs])

    api_key = os.getenv("OPENROUTER_API_KEY")

    if USE_GENERATION and api_key and context:
        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{query}\n"
        )
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GEN_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            answer_text = data["choices"][0]["message"]["content"].strip()
            return {"answer": answer_text, "contexts": best_docs}
        except Exception as e:
            logging.error(f"OpenRouter generation failed: {e}")

    return {"answer": _fallback_answer(query, best_docs), "contexts": best_docs}
