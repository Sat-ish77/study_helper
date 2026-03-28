-- Conversation tables for Study Helper v2
-- Stores saved chat conversations with message history

CREATE TABLE IF NOT EXISTS sh_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    messages JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for faster user-specific queries
CREATE INDEX IF NOT EXISTS idx_sh_conversations_user ON sh_conversations(user_id);

-- Index for sorting by updated time
CREATE INDEX IF NOT EXISTS idx_sh_conversations_updated ON sh_conversations(updated_at DESC);

-- RLS (Row Level Security) policies
ALTER TABLE sh_conversations ENABLE ROW LEVEL SECURITY;

-- Users can only access their own conversations
CREATE POLICY "Users can view own conversations" ON sh_conversations
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own conversations" ON sh_conversations
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update own conversations" ON sh_conversations
    FOR UPDATE USING (auth.uid()::text = user_id);

CREATE POLICY "Users can delete own conversations" ON sh_conversations
    FOR DELETE USING (auth.uid()::text = user_id);

-- Grant permissions
GRANT ALL ON sh_conversations TO authenticated;
GRANT ALL ON sh_conversations TO service_role;
