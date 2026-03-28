-- Learn More cache table for YouTube and Wikimedia content
-- Run this in Supabase SQL editor

CREATE TABLE IF NOT EXISTS sh_learn_more_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic TEXT NOT NULL,
    youtube_videos JSONB,
    images JSONB,
    cached_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '30 days'),
    UNIQUE(topic)
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_sh_learn_more_cache_topic ON sh_learn_more_cache(topic);
CREATE INDEX IF NOT EXISTS idx_sh_learn_more_cache_expires ON sh_learn_more_cache(expires_at);

-- RLS policy (if needed)
-- ALTER TABLE sh_learn_more_cache ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Users can view all cached content" ON sh_learn_more_cache FOR SELECT USING (true);
-- CREATE POLICY "Service role can manage cache" ON sh_learn_more_cache FOR ALL USING (auth.role() = 'service_role');
