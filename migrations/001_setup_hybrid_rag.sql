-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name TEXT,
    drive_file_id TEXT UNIQUE,
    mime_type TEXT,
    checksum TEXT,
    modified_time TIMESTAMP WITH TIME ZONE,
    course_code TEXT,
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create parent_chunks table
CREATE TABLE IF NOT EXISTS parent_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    section_title TEXT,
    content TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create child_chunks table
CREATE TABLE IF NOT EXISTS child_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id UUID REFERENCES parent_chunks(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT,
    embedding vector(1536), -- text-embedding-3-small dim is 1536
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index for vector search
CREATE INDEX IF NOT EXISTS child_chunks_embedding_idx ON child_chunks USING hnsw (embedding vector_cosine_ops);
-- Create index for full text search
CREATE INDEX IF NOT EXISTS child_chunks_fts_idx ON child_chunks USING gin (fts);

-- Function for hybrid search
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1536),
    match_count INT,
    full_text_weight FLOAT DEFAULT 1.0,
    semantic_weight FLOAT DEFAULT 1.0,
    rrf_k INT DEFAULT 60
)
RETURNS TABLE (
    child_id UUID,
    parent_id UUID,
    child_content TEXT,
    parent_content TEXT,
    similarity FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH semantic_search AS (
        SELECT 
            child_chunks.id, 
            child_chunks.parent_id,
            child_chunks.content,
            child_chunks.metadata,
            RANK() OVER (ORDER BY child_chunks.embedding <=> query_embedding) as rank,
            1 - (child_chunks.embedding <=> query_embedding) as similarity_score
        FROM child_chunks
        ORDER BY child_chunks.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    keyword_search AS (
        SELECT 
            child_chunks.id, 
            RANK() OVER (ORDER BY ts_rank(child_chunks.fts, websearch_to_tsquery('english', query_text)) DESC) as rank
        FROM child_chunks
        WHERE child_chunks.fts @@ websearch_to_tsquery('english', query_text)
        ORDER BY rank
        LIMIT match_count * 2
    ),
    hybrid_results AS (
        SELECT
            COALESCE(ss.id, ks.id) as id,
            COALESCE(ss.parent_id, child_chunks.parent_id) as parent_id,
            COALESCE(ss.content, child_chunks.content) as child_content,
            COALESCE(ss.metadata, child_chunks.metadata) as metadata,
            COALESCE(ss.similarity_score, 0.0) as similarity,
            (COALESCE(semantic_weight / (ss.rank + rrf_k), 0.0) +
             COALESCE(full_text_weight / (ks.rank + rrf_k), 0.0)) as rrf_score
        FROM semantic_search ss
        FULL OUTER JOIN keyword_search ks ON ss.id = ks.id
        LEFT JOIN child_chunks ON child_chunks.id = COALESCE(ss.id, ks.id)
        ORDER BY rrf_score DESC
        LIMIT match_count
    )
    SELECT
        hr.id as child_id,
        hr.parent_id,
        hr.child_content,
        pc.content as parent_content,
        hr.similarity,
        hr.metadata
    FROM hybrid_results hr
    LEFT JOIN parent_chunks pc ON hr.parent_id = pc.id;
END;
$$;
