'use client'

/**
 * Notes list panel component (left sidebar)
 *
 * Input: onSelectNote callback, currentNoteId, onClose callback
 * Output: List of notes with search, create, pin, and delete functionality
 *
 * Called by: MainApp component
 * Calls: supabase.from('notes').select, insert, update (pin/unpin, soft delete)
 */

import { useEffect, useState } from 'react'
import { createClient } from '@/lib/supabase-client'
import type { Note } from '@/types'

interface NotesPanelProps {
  onSelectNote: (noteId: string) => void
  currentNoteId: string | null
  onClose: () => void
}

export function NotesPanel({ onSelectNote, currentNoteId, onClose }: NotesPanelProps) {
  const [notes, setNotes] = useState<Note[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const supabase = createClient()

  /**
   * Load notes from database
   */
  useEffect(() => {
    loadNotes()
  }, [])

  const loadNotes = async () => {
    setIsLoading(true)
    try {
      const { data, error } = await supabase
        .from('notes')
        .select('*')
        .eq('is_deleted', false)
        .order('updated_at', { ascending: false }) as { data: any[]; error: any }

      if (error) throw error

      // Sort: pinned notes first (by pinned_at desc), then by updated_at desc
      const sorted = (data || []).sort((a: any, b: any) => {
        if (a.is_pinned && !b.is_pinned) return -1
        if (!a.is_pinned && b.is_pinned) return 1
        if (a.is_pinned && b.is_pinned) {
          return new Date(b.pinned_at!).getTime() - new Date(a.pinned_at!).getTime()
        }
        return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
      })

      setNotes(sorted)
    } catch (error) {
      console.error('Error loading notes:', error)
    } finally {
      setIsLoading(false)
    }
  }

  /**
   * Create new note
   */
  const handleCreateNote = async () => {
    try {
      // Get current user
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        console.error('No user found')
        return
      }

      const { data, error } = await supabase
        .from('notes')
        .insert({
          user_id: user.id,
          title: 'Untitled',
          content: '',
          plain_text: '',
        } as any)
        .select()
        .single() as { data: any; error: any }

      if (error) throw error
      setNotes(prev => [data, ...prev])
      onSelectNote(data?.id)
    } catch (error) {
      console.error('Error creating note:', error)
    }
  }

  /**
   * Pin/Unpin note
   */
  const handlePinNote = async (noteId: string, currentPinState: boolean, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent note selection when clicking pin

    try {
      const newPinState = !currentPinState
      const { error } = await (supabase
        .from('notes') as any)
        .update({
          is_pinned: newPinState,
          pinned_at: newPinState ? new Date().toISOString() : undefined
        })
        .eq('id', noteId)

      if (error) throw error

      // Update local state
      setNotes(prev => {
        const updated = prev.map(note =>
          note.id === noteId
            ? { ...note, is_pinned: newPinState, pinned_at: newPinState ? new Date().toISOString() : undefined }
            : note
        )
        // Sort: pinned notes first (by pinned_at desc), then by updated_at desc
        return updated.sort((a, b) => {
          if (a.is_pinned && !b.is_pinned) return -1
          if (!a.is_pinned && b.is_pinned) return 1
          if (a.is_pinned && b.is_pinned) {
            return new Date(b.pinned_at!).getTime() - new Date(a.pinned_at!).getTime()
          }
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
        })
      })
    } catch (error) {
      console.error('Error pinning note:', error)
      alert('Failed to pin note. Please try again.')
    }
  }

  /**
   * Delete note (soft delete)
   */
  const handleDeleteNote = async (noteId: string, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent note selection when clicking delete

    // Confirmation dialog
    const confirmed = window.confirm('Are you sure you want to delete this note? This action cannot be undone.')
    if (!confirmed) return

    try {
      const { error } = await (supabase
        .from('notes') as any)
        .update({ is_deleted: true })
        .eq('id', noteId)

      if (error) throw error

      // Remove note from local state
      setNotes(prev => prev.filter(note => note.id !== noteId))

      // If the deleted note was selected, clear selection
      if (currentNoteId === noteId) {
        onSelectNote('')
      }
    } catch (error) {
      console.error('Error deleting note:', error)
      alert('Failed to delete note. Please try again.')
    }
  }

  const filteredNotes = notes.filter(note =>
    note.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    note.plain_text.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <h2 className="font-medium text-text-primary">Notes</h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-border rounded transition-colors"
          title="Close panel"
        >
          <svg className="w-4 h-4 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Search */}
      <div className="p-3">
        <input
          type="text"
          placeholder="Search notes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 text-sm border border-border rounded focus:outline-none focus:ring-2 focus:ring-accent-blue"
        />
      </div>

      {/* New Note Button */}
      <div className="px-3 pb-3">
        <button
          onClick={handleCreateNote}
          className="w-full px-3 py-2 text-sm bg-accent-blue text-white rounded hover:bg-blue-600 transition-colors"
        >
          + New Note
        </button>
      </div>

      {/* Notes List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {isLoading ? (
          <div className="p-4 text-sm text-text-muted text-center">Loading notes...</div>
        ) : filteredNotes.length === 0 ? (
          <div className="p-4 text-sm text-text-muted text-center">
            {searchQuery ? 'No notes found' : 'No notes yet. Create one to get started!'}
          </div>
        ) : (
          <div className="space-y-1 p-2">
            {filteredNotes.map(note => (
              <div key={note.id} className="relative group">
                <button
                  onClick={() => onSelectNote(note.id)}
                  className={`w-full text-left px-3 py-2 pb-6 rounded transition-colors ${
                    currentNoteId === note.id
                      ? 'bg-accent-blue text-white'
                      : 'hover:bg-border-light text-text-primary'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-medium text-sm truncate flex-1">{note.title}</div>
                  </div>
                  <div className={`text-xs truncate mt-1 ${
                    currentNoteId === note.id ? 'text-white/80' : 'text-text-secondary'
                  }`}>
                    {note.plain_text && note.plain_text.trim() !== '' ? note.plain_text : (note.title !== 'Untitled' ? note.title : 'Empty note')}
                  </div>
                </button>

                {/* Favorite button - top right corner */}
                <button
                  onClick={(e) => handlePinNote(note.id, note.is_pinned, e)}
                  className={`absolute top-1.5 right-1.5 p-1 rounded transition-all ${
                    note.is_pinned
                      ? 'opacity-100'
                      : 'opacity-0 group-hover:opacity-100'
                  }`}
                  title={note.is_pinned ? 'Remove from favorites' : 'Add to favorites'}
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24">
                    <path
                      d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"
                      fill={note.is_pinned ? '#FBBF24' : 'none'}
                      stroke={note.is_pinned ? '#F59E0B' : '#9CA3AF'}
                      strokeWidth={1.5}
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>

                {/* Delete button - bottom right corner */}
                <button
                  onClick={(e) => handleDeleteNote(note.id, e)}
                  className={`absolute bottom-1 right-1 p-1.5 rounded transition-all opacity-0 group-hover:opacity-100 ${
                    currentNoteId === note.id
                      ? 'hover:bg-white/20 text-white'
                      : 'hover:bg-red-100 text-text-muted hover:text-red-600'
                  }`}
                  title="Delete note"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}
