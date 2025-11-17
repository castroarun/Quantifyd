'use client'

/**
 * Rich text editor component using Tiptap
 *
 * Input: noteId (optional), userId
 * Output: Tiptap editor with toolbar and auto-save
 *
 * Called by: MainApp component
 * Calls: Tiptap extensions, auto-save logic
 */

import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Link from '@tiptap/extension-link'
import Placeholder from '@tiptap/extension-placeholder'
import Underline from '@tiptap/extension-underline'
import { useEffect, useState, useCallback } from 'react'
import { createClient } from '@/lib/supabase-client'
import type { Note } from '@/types'

interface EditorProps {
  noteId: string | null
  userId: string
}

export function Editor({ noteId, userId }: EditorProps) {
  const [note, setNote] = useState<Note | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const [lastSaved, setLastSaved] = useState<string | null>(null)
  const [saveTimeout, setSaveTimeout] = useState<NodeJS.Timeout | null>(null)
  const [editableTitle, setEditableTitle] = useState<string>('Untitled')
  const supabase = createClient()

  /**
   * Initialize Tiptap editor with extensions and keyboard shortcuts
   */
  const editor = useEditor({
    immediatelyRender: false,
    extensions: [
      StarterKit.configure({
        heading: {
          levels: [1, 2, 3],
        },
      }),
      Link.configure({
        openOnClick: false,
      }),
      Underline,
      Placeholder.configure({
        placeholder: 'Start typing your note...',
      }),
    ],
    content: '',
    editorProps: {
      attributes: {
        class: 'prose prose-sm sm:prose lg:prose-lg xl:prose-xl focus:outline-none min-h-full p-8',
      },
      handleKeyDown: (view, event) => {
        const { state } = view
        const { from, to } = state.selection
        const hasSelection = from !== to

        // Ctrl+B - Bold
        if (event.ctrlKey && event.key === 'b') {
          if (hasSelection) {
            event.preventDefault()
            editor?.chain().focus().toggleBold().run()
            return true
          }
          return false // Allow default Windows behavior
        }

        // Ctrl+I - Italic
        if (event.ctrlKey && event.key === 'i') {
          if (hasSelection) {
            event.preventDefault()
            editor?.chain().focus().toggleItalic().run()
            return true
          }
          return false
        }

        // Ctrl+U - Underline
        if (event.ctrlKey && event.key === 'u') {
          if (hasSelection) {
            event.preventDefault()
            editor?.chain().focus().toggleUnderline().run()
            return true
          }
          return false
        }

        // Ctrl+H - Header
        if (event.ctrlKey && event.key === 'h') {
          if (hasSelection) {
            event.preventDefault()
            editor?.chain().focus().toggleHeading({ level: 2 }).run()
            return true
          }
          return false
        }

        // Tab - Toggle bullet list or sink list item (nest)
        if (event.key === 'Tab' && !event.ctrlKey && !event.altKey && !event.metaKey) {
          event.preventDefault()

          // If already in a list item
          if (editor?.isActive('listItem')) {
            if (event.shiftKey) {
              // Shift+Tab - lift (un-nest) or exit list if at top level
              const lifted = editor?.chain().focus().liftListItem('listItem').run()
              if (!lifted) {
                // If can't lift, toggle off the list
                editor?.chain().focus().toggleBulletList().run()
              }
            } else {
              // Tab - sink (nest deeper)
              editor?.chain().focus().sinkListItem('listItem').run()
            }
            return true
          }

          // Not in a list - create a bullet list
          editor?.chain().focus().toggleBulletList().run()
          return true
        }

        return false
      },
    },
    onUpdate: ({ editor }) => {
      // Trigger auto-save on content change
      handleAutoSave(editor.getHTML(), editor.getText())
    },
  })

  /**
   * Generate a smart title from note content
   *
   * Input: plainText of the note
   * Output: Generated title string (does NOT modify note content)
   *
   * Called by: saveNote when title is empty
   * Calls: None
   */
  const generateTitle = (plainText: string): string => {
    if (!plainText || plainText.trim() === '') {
      return 'Untitled'
    }

    // Extract first meaningful line (up to 100 chars) without removing it from content
    const lines = plainText.split('\n')
    for (const line of lines) {
      const trimmed = line.trim()
      if (trimmed.length > 0) {
        return trimmed.substring(0, 100)
      }
    }

    return 'Untitled'
  }

  /**
   * Auto-save logic with 2-second debounce
   *
   * Input: content (HTML), plainText
   * Output: Saves note to database after 2 seconds of inactivity
   *
   * Called by: Editor onUpdate
   * Calls: saveNote
   */
  const handleAutoSave = useCallback(
    (content: string, plainText: string) => {
      if (saveTimeout) {
        clearTimeout(saveTimeout)
      }

      const timeout = setTimeout(async () => {
        await saveNote(content, plainText)
      }, 2000)

      setSaveTimeout(timeout)
    },
    [saveTimeout, noteId]
  )

  /**
   * Save note to database
   *
   * Input: content (HTML), plainText
   * Output: Updates or creates note in Supabase
   *
   * Called by: Auto-save timeout
   * Calls: supabase.from('notes').upsert, generateTitle
   */
  const saveNote = async (content: string, plainText: string) => {
    setIsSaving(true)

    try {
      // Only auto-generate title if user hasn't set one or it's still "Untitled"
      let finalTitle = editableTitle
      if (!finalTitle || finalTitle.trim() === '' || finalTitle === 'Untitled') {
        // Auto-generate from content for convenience, but keep content intact
        finalTitle = generateTitle(plainText)
      }

      const noteData = {
        id: noteId || undefined,
        user_id: userId,
        title: finalTitle,
        content, // Content is NEVER modified
        plain_text: plainText,
        updated_at: new Date().toISOString(),
      }

      const { data, error } = await supabase
        .from('notes')
        .upsert(noteData)
        .select()
        .single()

      if (error) throw error

      setNote(data)
      // Update editable title with saved value
      if (editableTitle !== data.title) {
        setEditableTitle(data.title)
      }
      setLastSaved(new Date().toLocaleTimeString())
    } catch (error) {
      console.error('Error saving note:', error)
    } finally {
      setIsSaving(false)
    }
  }

  /**
   * Load note content when noteId changes
   */
  useEffect(() => {
    if (!noteId) {
      editor?.commands.setContent('')
      setNote(null)
      setEditableTitle('Untitled')
      return
    }

    const loadNote = async () => {
      const { data, error } = await supabase
        .from('notes')
        .select('*')
        .eq('id', noteId)
        .single()

      if (error) {
        console.error('Error loading note:', error)
        return
      }

      setNote(data)
      setEditableTitle(data.title || 'Untitled')
      editor?.commands.setContent(data.content || '')
    }

    loadNote()
  }, [noteId, editor, supabase])

  if (!editor) {
    return <div className="flex items-center justify-center h-full">Loading editor...</div>
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Save status indicator - minimal top bar */}
      <div className="flex items-center justify-end px-4 py-2 border-b border-border bg-background">
        <div className="flex items-center gap-2 text-sm text-text-muted">
          {isSaving ? (
            <>
              <div className="w-4 h-4 border-2 border-text-muted border-t-transparent rounded-full animate-spin" />
              Saving...
            </>
          ) : lastSaved ? (
            <>
              <svg className="w-4 h-4 text-accent-green" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              Saved at {lastSaved}
            </>
          ) : null}
        </div>
      </div>

      {/* Editable Note Title */}
      {noteId && (
        <div className="px-8 pt-6 pb-2 bg-background">
          <input
            type="text"
            value={editableTitle}
            onChange={(e) => setEditableTitle(e.target.value)}
            onBlur={() => {
              // Trigger save when title is edited
              if (editor) {
                handleAutoSave(editor.getHTML(), editor.getText())
              }
            }}
            className="w-full text-2xl font-bold text-text-primary bg-transparent border-none outline-none focus:ring-0 placeholder-text-muted"
            placeholder="Untitled Note"
          />
        </div>
      )}

      {/* Editor Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin bg-background">
        <EditorContent editor={editor} className="h-full" />
      </div>
    </div>
  )
}
