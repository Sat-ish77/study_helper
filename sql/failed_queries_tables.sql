-- Failed queries table for Study Helper v2
-- Tracks RAG failures for debugging

CREATE TABLE IF NOT EXISTS sh_failed_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    top_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for faster user-specific queries
CREATE INDEX IF NOT EXISTS idx_sh_failed_queries_user ON sh_failed_queries(user_id);

-- Index for sorting by time
CREATE INDEX IF NOT EXISTS idx_sh_failed_queries_created ON sh_failed_queries(created_at DESC);

-- RLS (Row Level Security) policies
ALTER TABLE sh_failed_queries ENABLE ROW LEVEL SECURITY;

-- Users can only view their own failed queries
CREATE POLICY "Users can view own failed queries" ON sh_failed_queries
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own failed queries" ON sh_failed_queries
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

-- Grant permissions
GRANT ALL ON sh_failed_queries TO authenticated;
GRANT ALL ON sh_failed_queries TO service_role;
