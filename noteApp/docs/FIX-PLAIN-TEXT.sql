-- ============================================================================
-- Fix: Update plain_text for existing notes
-- ============================================================================
-- Run this in Supabase SQL Editor to populate plain_text for existing notes

-- This will extract text from HTML content and populate plain_text
-- For notes where plain_text is empty or null

UPDATE public.notes
SET plain_text =
  -- Strip HTML tags and decode common entities
  regexp_replace(
    regexp_replace(content, '<[^>]+>', ' ', 'g'),  -- Remove HTML tags
    '\s+', ' ', 'g'  -- Normalize whitespace
  ),
  updated_at = updated_at  -- Don't change the updated timestamp
WHERE plain_text IS NULL OR plain_text = '' OR trim(plain_text) = '';
