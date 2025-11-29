'use client'

/**
 * Main application component with editor and collapsible panels
 *
 * Input: user object from Supabase auth
 * Output: Complete note-taking interface
 *
 * Called by: app/page.tsx when user is authenticated
 * Calls: Editor, NotesPanel, RightPanel, UserMenu
 */

import { useState, useRef } from 'react'
import { Editor } from './Editor'
import { NotesPanel } from './NotesPanel'
import { RightPanel } from './RightPanel'
import { UserMenu } from './Auth'
import { useTheme } from '@/ui/contexts/ThemeContext'
import { createClient } from '@/lib/supabase-client'
import type { PanelState } from '@/types'

interface MainAppProps {
  user: any
}

export function MainApp({ user }: MainAppProps) {
  const [panels, setPanels] = useState<PanelState>({
    left: true,
    right: true,
  })
  const [currentNoteId, setCurrentNoteId] = useState<string | null>(null)
  const [notesPanelKey, setNotesPanelKey] = useState(0)
  const { theme, toggleTheme } = useTheme()
  const supabase = createClient()

  /**
   * Create a new note with template content
   */
  const handleCreateFromTemplate = async (title: string, content: string, plainText: string) => {
    try {
      const { data, error } = await supabase
        .from('notes')
        .insert({
          user_id: user.id,
          title,
          content,
          plain_text: plainText,
        } as any)
        .select()
        .single() as { data: any; error: any }

      if (error) throw error

      // Select the new note
      setCurrentNoteId(data?.id)
      // Refresh the notes panel
      setNotesPanelKey(prev => prev + 1)
    } catch (error) {
      console.error('Error creating note from template:', error)
    }
  }

  /**
   * Toggle panel visibility
   *
   * Input: panel name ('left' or 'right')
   * Output: Updates panel state
   *
   * Called by: Panel collapse buttons
   * Calls: setPanels
   */
  const togglePanel = (panel: 'left' | 'right') => {
    setPanels(prev => ({
      ...prev,
      [panel]: !prev[panel],
    }))
  }

  return (
    <div className="h-full w-full flex flex-col bg-background">
      {/* Top header bar */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border bg-background">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-light text-text-primary">NoteApp</h1>
        </div>
        <div className="flex items-center gap-3">
          {/* Dark Mode Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg hover:bg-border-light transition-colors"
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? (
              /* Sun icon for light mode */
              <svg className="w-5 h-5 text-text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            ) : (
              /* Moon icon for dark mode */
              <svg className="w-5 h-5 text-text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            )}
          </button>
          <UserMenu user={user} />
        </div>
      </header>

      {/* Main content area with 3 sections */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Panel - Notes List */}
        {panels.left && (
          <div className="w-[280px] bg-border-light border-r border-border flex flex-col">
            <NotesPanel
              key={notesPanelKey}
              onSelectNote={setCurrentNoteId}
              currentNoteId={currentNoteId}
              onClose={() => togglePanel('left')}
            />
          </div>
        )}

        {/* Center Panel - Editor (always visible, takes remaining space) */}
        <div className="flex-1 flex flex-col">
          {!panels.left && (
            <button
              onClick={() => togglePanel('left')}
              className="absolute top-16 left-2 p-2 bg-background border border-border rounded hover:bg-border-light transition-colors z-10"
              title="Show notes list"
            >
              <svg
                className="w-4 h-4 text-text-secondary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </button>
          )}

          <Editor noteId={currentNoteId} userId={user.id} />

        </div>

        {/* Right Panel - Templates */}
        <RightPanel onCreateFromTemplate={handleCreateFromTemplate} />
      </main>
    </div>
  )
}
