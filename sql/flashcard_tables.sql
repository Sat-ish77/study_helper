-- Flashcards tables for SM-2 spaced repetition
-- Run these in Supabase SQL editor

-- Main flashcards table
CREATE TABLE IF NOT EXISTS sh_flashcards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    source TEXT DEFAULT 'manual', -- manual, qa, generated
    context TEXT,
    
    -- SM-2 Algorithm fields
    ease_factor FLOAT DEFAULT 2.5,
    repetition_interval INTEGER DEFAULT 1,
    repetitions INTEGER DEFAULT 0,
    next_review TIMESTAMPTZ DEFAULT NOW(),
    last_review TIMESTAMPTZ,
    last_quality INTEGER,
    review_time_ms INTEGER,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sh_flashcards_user_id ON sh_flashcards(user_id);
CREATE INDEX IF NOT EXISTS idx_sh_flashcards_next_review ON sh_flashcards(next_review);
CREATE INDEX IF NOT EXISTS idx_sh_flashcards_created_at ON sh_flashcards(created_at);

-- RLS policies (if needed)
-- ALTER TABLE sh_flashcards ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Users can manage their flashcards" ON sh_flashcards FOR ALL USING (auth.uid()::text = user_id);

-- Update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_flashcards_updated_at 
    BEFORE UPDATE ON sh_flashcards 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();
