-- ============================================================================
-- Migration: Add Pinned Notes and Dark Mode Support
-- ============================================================================
-- Run this in your Supabase SQL editor to add new features

-- Add pin support to notes table
ALTER TABLE public.notes
ADD COLUMN IF NOT EXISTS is_pinned BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS pinned_at TIMESTAMPTZ;

-- Create index for pinned notes (for efficient sorting)
CREATE INDEX IF NOT EXISTS notes_pinned_idx ON public.notes(is_pinned, pinned_at DESC)
  WHERE is_pinned = TRUE;

-- Add dark mode preference to profiles table
ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS theme_preference TEXT DEFAULT 'light' CHECK (theme_preference IN ('light', 'dark'));

-- Comment explaining the new columns
COMMENT ON COLUMN public.notes.is_pinned IS 'Whether note is pinned to top of list';
COMMENT ON COLUMN public.notes.pinned_at IS 'When note was pinned (for sorting pinned notes)';
COMMENT ON COLUMN public.profiles.theme_preference IS 'User preferred theme (light or dark)';
