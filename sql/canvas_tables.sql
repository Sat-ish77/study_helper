-- Canvas integration tables
-- Run these in Supabase SQL editor

-- Canvas iCal cache (30-minute TTL)
CREATE TABLE IF NOT EXISTS sh_canvas_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key TEXT NOT NULL UNIQUE,
    events JSONB,
    cached_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '30 minutes')
);

-- Dismissed events (so users can hide events they don't care about)
CREATE TABLE IF NOT EXISTS sh_dismissed_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    dismissed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, event_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sh_canvas_cache_key ON sh_canvas_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_sh_canvas_cache_expires ON sh_canvas_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_sh_dismissed_events_user ON sh_dismissed_events(user_id);

-- RLS policies (if needed)
-- ALTER TABLE sh_canvas_cache ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Service role can manage cache" ON sh_canvas_cache FOR ALL USING (auth.role() = 'service_role');

-- ALTER TABLE sh_dismissed_events ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Users can manage their dismissed events" ON sh_dismissed_events FOR ALL USING (auth.uid()::text = user_id);
