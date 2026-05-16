from __future__ import annotations

from typing import Callable, Dict, List, Optional, Any
import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client # type: ignore

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class VectorStore:
    def __init__(self) -> None:
        self.supabase: Optional[Client] = None
        if SUPABASE_URL and SUPABASE_KEY:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        else:
            print("Warning: SUPABASE_URL or SUPABASE_KEY not found in environment.")

    def add_chunks(
        self,
        chunks_hierarchy: List[Dict[str, Any]],
        embed_fn: Callable[[str], List[float]],
        metadata: Dict[str, str] | None = None,
    ) -> int:
        added = 0
        base_meta = metadata or {}
        
        file_metadata = {
            "drive_file_id": base_meta.get("drive_file_id", f"local-{base_meta.get('filename', 'unknown')}-{datetime.now().timestamp()}"),
            "file_name": base_meta.get("filename", "unknown"),
            "mime_type": base_meta.get("mime_type", "application/octet-stream"),
            "checksum": base_meta.get("checksum", ""),
            "modified_time": base_meta.get("modified_time", datetime.now().isoformat()),
            "course_code": base_meta.get("course_code", "GENERAL"),
        }

        if not self.supabase:
            raise RuntimeError("Supabase client not initialized. Check your environment variables.")

        try:
            # 1. Check if document with same checksum exists to avoid re-indexing
            if file_metadata["checksum"]:
                existing = self.supabase.table("documents").select("id").eq("checksum", file_metadata["checksum"]).execute()
                if existing.data:
                    print(f"Document with checksum {file_metadata['checksum']} already indexed. Skipping.")
                    return 0

            # 2. Upsert document
            doc_record = {
                "drive_file_id": file_metadata["drive_file_id"],
                "file_name": file_metadata["file_name"],
                "mime_type": file_metadata["mime_type"],
                "checksum": file_metadata["checksum"],
                "modified_time": file_metadata["modified_time"],
                "course_code": file_metadata["course_code"],
                "indexed_at": datetime.now().isoformat(),
            }
            
            response = self.supabase.table("documents").upsert(
                doc_record, on_conflict="drive_file_id"
            ).execute()
            
            if not response.data:
                raise RuntimeError("Failed to upsert document to Supabase")
                
            doc_id = response.data[0]["id"]
            
            # 2. Iterate hierarchy
            for parent in chunks_hierarchy:
                # Insert parent chunk
                parent_res = self.supabase.table("parent_chunks").insert({
                    "document_id": doc_id,
                    "section_title": parent["parent_title"],
                    "content": parent["parent_content"],
                    "metadata": {"source": file_metadata["file_name"]}
                }).execute()
                
                if not parent_res.data:
                    continue
                parent_id = parent_res.data[0]["id"]
                
                # Prepare and insert child chunks
                child_records = []
                for child_text in parent["children"]:
                    child_text = child_text.strip()
                    if not child_text:
                        continue
                    
                    vector = embed_fn(child_text)
                    
                    child_records.append({
                        "parent_id": parent_id,
                        "document_id": doc_id,
                        "content": child_text,
                        "embedding": vector,
                        "metadata": {
                            "source": file_metadata["file_name"],
                            "section": parent["parent_title"]
                        }
                    })
                    added += 1
                
                if child_records:
                    self.supabase.table("child_chunks").insert(child_records).execute()

        except Exception as e:
            print(f"Error in add_chunks: {e}")
            raise e

        return added

    def search(self, query_text: str, query_vector: List[float], top_k: int = 20) -> List[dict]:
        if top_k <= 0 or not self.supabase:
            return []

        try:
            # Using Supabase rpc for hybrid search
            rpc_params = {
                "query_text": query_text,
                "query_embedding": query_vector,
                "match_count": top_k,
                "full_text_weight": 1.0,
                "semantic_weight": 1.0,
                "rrf_k": 60
            }
            
            response = self.supabase.rpc("hybrid_search", rpc_params).execute()
            
            results = []
            if response.data:
                for row in response.data:
                    results.append({
                        "child_text": row.get("child_content", ""),
                        "parent_text": row.get("parent_content", ""),
                        "score": row.get("similarity", 0.0),
                        "metadata": row.get("metadata", {}),
                    })
            return results
        except Exception as e:
            print(f"Error in search: {e}")
            return []

    def get_stats(self) -> dict:
        if not self.supabase:
            return {"error": "Supabase not initialized"}
        
        try:
            doc_count = self.supabase.table("documents").select("id", count="exact").execute().count
            parent_count = self.supabase.table("parent_chunks").select("id", count="exact").execute().count
            child_count = self.supabase.table("child_chunks").select("id", count="exact").execute().count
            return {
                "total_documents": doc_count,
                "total_parent_chunks": parent_count,
                "total_chunks": child_count
            }
        except Exception as e:
            return {"error": str(e)}

default_store = VectorStore()

def add_chunks(chunks_hierarchy, embed_fn, metadata=None):
    return default_store.add_chunks(chunks_hierarchy, embed_fn, metadata=metadata)

def search(query_text, query_vector, top_k=20):
    return default_store.search(query_text, query_vector, top_k=top_k)
