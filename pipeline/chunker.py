from __future__ import annotations
import re
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

# Ensure punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

# Use the token encoding for the chosen embedding model
enc = tiktoken.encoding_for_model("text-embedding-3-small")

def token_count(text: str) -> int:
    return len(enc.encode(text))

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str) -> List[Dict[str, Any]]:
    """
    Implements a best-in-class modern RAG chunking algorithm.
    - Creates Semantic Parents (by topic/headings).
    - Splits parents exceeding max token counts.
    - Uses an NLP Sliding Sentence Window for child chunks.
    - Accurately groups by Token count instead of Word count.
    - Embeds Contextual Headers directly into child chunks for retrieval.
    """
    PARENT_MAX_TOKENS = 2000
    CHILD_MAX_TOKENS = 400
    OVERLAP_SENTENCES = 2

    # Step 1: Split into raw semantic sections based on Markdown Headings
    sections = re.split(r'(?m)^(#+\s+.*)\n', text)
    
    raw_parents = []
    if sections and not sections[0].strip().startswith('#'):
        content = sections[0].strip()
        if content:
            raw_parents.append({
                "title": "Introduction",
                "content": content
            })
    
    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        # Clean title by removing markdown '#'
        clean_title = re.sub(r'^#+\s+', '', title).strip()
        
        content = sections[i+1].strip() if i+1 < len(sections) else ""
        raw_parents.append({
            "title": clean_title,
            "content": f"{title}\n\n{content}".strip()
        })
    
    # Step 2: Enforce Parent Token Limits (Rule B - Parent Too Large)
    parents = []
    for rp in raw_parents:
        title = rp["title"]
        content = rp["content"]
        if token_count(content) <= PARENT_MAX_TOKENS:
            parents.append(rp)
        else:
            # If the section is a massive textbook chapter, chunk it further by sentences
            sentences = sent_tokenize(content)
            current_parent_sents = []
            current_tokens = 0
            part = 1
            
            for sent in sentences:
                sent = _normalize_whitespace(sent)
                if not sent: continue
                t_count = token_count(sent)
                
                if current_tokens + t_count > PARENT_MAX_TOKENS and current_parent_sents:
                    parents.append({
                        "title": f"{title} (Part {part})",
                        "content": " ".join(current_parent_sents)
                    })
                    part += 1
                    current_parent_sents = [sent]
                    current_tokens = t_count
                else:
                    current_parent_sents.append(sent)
                    current_tokens += t_count
                    
            if current_parent_sents:
                parents.append({
                    "title": f"{title} (Part {part})" if part > 1 else title,
                    "content": " ".join(current_parent_sents)
                })

    # Step 3: Semantic Child Chunking (Sentence sliding window + Tokens + Context Headers)
    chunks_hierarchy = []
    
    for p in parents:
        parent_title = p["title"]
        parent_content = p["content"]
        
        if not parent_content:
            continue
            
        sentences = sent_tokenize(parent_content)
        
        small_chunks = []
        current_chunk_sents = []
        current_tokens = 0
        
        # Context header added to EVERY child chunk
        header = f"Section: {parent_title}\n\n"
        header_tokens = token_count(header)
        
        for sent in sentences:
            sent = _normalize_whitespace(sent)
            if not sent: continue
            
            t_count = token_count(sent)
            
            if current_tokens + t_count > (CHILD_MAX_TOKENS - header_tokens) and current_chunk_sents:
                chunk_text_body = " ".join(current_chunk_sents)
                small_chunks.append(f"{header}{chunk_text_body}")
                
                # Semantic Overlap: Keep the last N sentences fully intact
                overlap_sents = current_chunk_sents[-OVERLAP_SENTENCES:] if len(current_chunk_sents) > OVERLAP_SENTENCES else current_chunk_sents
                
                current_chunk_sents = overlap_sents + [sent]
                current_tokens = sum(token_count(s) for s in current_chunk_sents)
            else:
                current_chunk_sents.append(sent)
                current_tokens += t_count
                
        if current_chunk_sents:
            chunk_text_body = " ".join(current_chunk_sents)
            small_chunks.append(f"{header}{chunk_text_body}")
            
        chunks_hierarchy.append({
            "parent_title": parent_title,
            "parent_content": parent_content,
            "children": small_chunks
        })
        
    return chunks_hierarchy
