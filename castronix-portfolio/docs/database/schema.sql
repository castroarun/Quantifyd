-- ============================================
-- CASTRONIX PORTFOLIO DATABASE SCHEMA
-- Supabase PostgreSQL
-- ============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- PROJECTS TABLE
-- Stores portfolio projects with status
-- ============================================
CREATE TABLE projects (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  slug TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  tagline TEXT,
  status TEXT NOT NULL DEFAULT 'coming_soon'
    CHECK (status IN ('coming_soon', 'beta', 'live')),
  url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for status queries
CREATE INDEX idx_projects_status ON projects(status);

-- Insert initial projects
INSERT INTO projects (slug, name, tagline, status) VALUES
  ('reppit', 'REPPIT', 'Track your reps. Build your strength.', 'live'),
  ('noteapp', 'NoteApp', 'Simple, fast note-taking that stays out of your way.', 'live'),
  ('primmo', 'PRIMMO', 'Real estate investment analysis made intelligent.', 'coming_soon');

-- ============================================
-- WAITLIST TABLE
-- Stores email subscriptions for project launches
-- ============================================
CREATE TABLE waitlist (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT NOT NULL,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  notified BOOLEAN DEFAULT FALSE,
  subscribed_at TIMESTAMPTZ DEFAULT NOW(),
  notified_at TIMESTAMPTZ,

  -- Prevent duplicate subscriptions per project
  UNIQUE(email, project_id)
);

-- Indexes for efficient queries
CREATE INDEX idx_waitlist_project ON waitlist(project_id);
CREATE INDEX idx_waitlist_notified ON waitlist(notified) WHERE notified = FALSE;

-- ============================================
-- FUNCTION: Update timestamp trigger
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to projects table
CREATE TRIGGER projects_updated_at
  BEFORE UPDATE ON projects
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

-- ============================================
-- FUNCTION: Notify waitlist on status change
-- Calls edge function when project goes live
-- ============================================
CREATE OR REPLACE FUNCTION notify_on_launch()
RETURNS TRIGGER AS $$
BEGIN
  -- Only trigger when status changes TO 'live'
  IF NEW.status = 'live' AND OLD.status != 'live' THEN
    -- Call edge function via pg_net (Supabase extension)
    PERFORM net.http_post(
      url := current_setting('app.settings.supabase_url') || '/functions/v1/send-launch-notification',
      headers := jsonb_build_object(
        'Content-Type', 'application/json',
        'Authorization', 'Bearer ' || current_setting('app.settings.service_role_key')
      ),
      body := jsonb_build_object(
        'project_id', NEW.id,
        'project_name', NEW.name,
        'project_slug', NEW.slug
      )
    );
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Apply trigger to projects table
CREATE TRIGGER project_launch_notification
  AFTER UPDATE ON projects
  FOR EACH ROW
  EXECUTE FUNCTION notify_on_launch();

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================

-- Enable RLS
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE waitlist ENABLE ROW LEVEL SECURITY;

-- Projects: Anyone can read
CREATE POLICY "Projects are publicly viewable"
  ON projects FOR SELECT
  USING (true);

-- Waitlist: Anyone can insert (subscribe)
CREATE POLICY "Anyone can subscribe to waitlist"
  ON waitlist FOR INSERT
  WITH CHECK (true);

-- Waitlist: Only service role can read/update (for sending notifications)
CREATE POLICY "Service role can manage waitlist"
  ON waitlist FOR ALL
  USING (auth.role() = 'service_role');

-- ============================================
-- VIEWS
-- ============================================

-- View: Waitlist counts per project
CREATE VIEW waitlist_stats AS
SELECT
  p.slug,
  p.name,
  p.status,
  COUNT(w.id) as total_subscribers,
  COUNT(w.id) FILTER (WHERE w.notified = false) as pending_notifications
FROM projects p
LEFT JOIN waitlist w ON p.id = w.project_id
GROUP BY p.id, p.slug, p.name, p.status;
