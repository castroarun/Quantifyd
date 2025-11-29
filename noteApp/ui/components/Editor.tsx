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
import { Extension } from '@tiptap/core'
import BulletList from '@tiptap/extension-bullet-list'
import OrderedList from '@tiptap/extension-ordered-list'
import ListItem from '@tiptap/extension-list-item'
import { useEffect, useState, useCallback } from 'react'
import { createClient } from '@/lib/supabase-client'
import type { Note } from '@/types'

/**
 * Custom keyboard shortcuts extension
 * Alt+H for heading (Ctrl+H conflicts with browser history)
 */
const CustomKeyboardShortcuts = Extension.create({
  name: 'customKeyboardShortcuts',
  priority: 1000,

  addKeyboardShortcuts() {
    return {
      // Alt+H for heading (avoids browser shortcut conflict)
      'Alt-h': ({ editor }) => {
        return editor.commands.toggleHeading({ level: 2 })
      },
    }
  },
})

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
  const [wordCount, setWordCount] = useState(0)
  const [showShortcuts, setShowShortcuts] = useState(false)
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
        // Disable built-in list extensions to use custom ones with inline styles
        bulletList: false,
        orderedList: false,
        listItem: false,
      }),
      // Custom BulletList with inline styles to override Tailwind reset
      BulletList.configure({
        HTMLAttributes: {
          style: 'list-style-type: disc; padding-left: 1.5em; margin-left: 0.5em;',
        },
      }),
      // Custom OrderedList with inline styles
      OrderedList.configure({
        HTMLAttributes: {
          style: 'list-style-type: decimal; padding-left: 1.5em; margin-left: 0.5em;',
        },
      }),
      // Custom ListItem with inline styles
      ListItem.configure({
        HTMLAttributes: {
          style: 'display: list-item;',
        },
      }),
      Link.configure({
        openOnClick: false,
      }),
      Underline,
      Placeholder.configure({
        placeholder: 'Start typing your note...',
      }),
      CustomKeyboardShortcuts,
    ],
    content: '',
    editorProps: {
      attributes: {
        class: 'focus:outline-none min-h-full p-8 text-text-primary',
      },
    },
    onUpdate: ({ editor }) => {
      // Trigger auto-save on content change
      const text = editor.getText()
      handleAutoSave(editor.getHTML(), text)
      // Update word count
      const words = text.trim().split(/\s+/).filter(word => word.length > 0)
      setWordCount(words.length)
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
    // Don't save if there's no noteId (new unsaved note)
    if (!noteId) {
      return
    }

    // Don't save if content is empty or only whitespace
    const trimmedText = plainText.trim()
    if (!trimmedText || trimmedText === '') {
      return
    }

    setIsSaving(true)

    try {
      // Only auto-generate title if user hasn't set one or it's still "Untitled"
      let finalTitle = editableTitle
      if (!finalTitle || finalTitle.trim() === '' || finalTitle === 'Untitled') {
        // Auto-generate from content for convenience, but keep content intact
        finalTitle = generateTitle(plainText)
      }

      const noteData = {
        id: noteId,
        user_id: userId,
        title: finalTitle,
        content, // Content is NEVER modified
        plain_text: plainText,
        updated_at: new Date().toISOString(),
      }

      const { data, error } = await supabase
        .from('notes')
        .upsert(noteData as any)
        .select()
        .single() as { data: any; error: any }

      if (error) throw error

      setNote(data)
      // Update editable title with saved value
      if (data && editableTitle !== data.title) {
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
      // Focus editor even when no note is selected
      setTimeout(() => {
        editor?.commands.focus()
      }, 100)
      return
    }

    const loadNote = async () => {
      const { data, error } = await supabase
        .from('notes')
        .select('*')
        .eq('id', noteId)
        .single() as { data: any; error: any }

      if (error) {
        console.error('Error loading note:', error)
        return
      }

      setNote(data)
      setEditableTitle(data?.title || 'Untitled')
      editor?.commands.setContent(data?.content || '')

      // Auto-focus editor so user can immediately start typing
      setTimeout(() => {
        editor?.commands.focus()
      }, 100)
    }

    loadNote()
  }, [noteId, editor, supabase])

  if (!editor) {
    return <div className="flex items-center justify-center h-full">Loading editor...</div>
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Toolbar and Save status */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-background">
        {/* Formatting toolbar */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => editor.chain().focus().toggleBold().run()}
            className={`p-2 rounded hover:bg-border ${editor.isActive('bold') ? 'bg-border' : ''}`}
            title="Bold (Ctrl+B)"
          >
            <span className="font-bold text-sm">B</span>
          </button>
          <button
            onClick={() => editor.chain().focus().toggleItalic().run()}
            className={`p-2 rounded hover:bg-border ${editor.isActive('italic') ? 'bg-border' : ''}`}
            title="Italic (Ctrl+I)"
          >
            <span className="italic text-sm">I</span>
          </button>
          <button
            onClick={() => editor.chain().focus().toggleUnderline().run()}
            className={`p-2 rounded hover:bg-border ${editor.isActive('underline') ? 'bg-border' : ''}`}
            title="Underline (Ctrl+U)"
          >
            <span className="underline text-sm">U</span>
          </button>
          <button
            onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
            className={`p-2 rounded hover:bg-border ${editor.isActive('heading') ? 'bg-border' : ''}`}
            title="Heading (Alt+H)"
          >
            <span className="font-bold text-sm">H</span>
          </button>
          <div className="w-px h-5 bg-border mx-1" />
          <button
            onClick={() => editor.chain().focus().toggleBulletList().run()}
            className={`p-2 rounded hover:bg-border ${editor.isActive('bulletList') ? 'bg-border' : ''}`}
            title="Bullet List (- + Space)"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <circle cx="3" cy="5" r="2" />
              <rect x="7" y="4" width="11" height="2" rx="1" />
              <circle cx="3" cy="10" r="2" />
              <rect x="7" y="9" width="11" height="2" rx="1" />
              <circle cx="3" cy="15" r="2" />
              <rect x="7" y="14" width="11" height="2" rx="1" />
            </svg>
          </button>
          <button
            onClick={() => editor.chain().focus().toggleOrderedList().run()}
            className={`p-2 rounded hover:bg-border ${editor.isActive('orderedList') ? 'bg-border' : ''}`}
            title="Numbered List (1. + Space)"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <text x="1" y="6" fontSize="5" fontWeight="bold">1</text>
              <rect x="7" y="4" width="11" height="2" rx="1" />
              <text x="1" y="11" fontSize="5" fontWeight="bold">2</text>
              <rect x="7" y="9" width="11" height="2" rx="1" />
              <text x="1" y="16" fontSize="5" fontWeight="bold">3</text>
              <rect x="7" y="14" width="11" height="2" rx="1" />
            </svg>
          </button>
          <div className="w-px h-5 bg-border mx-1" />
          <button
            onClick={() => setShowShortcuts(!showShortcuts)}
            className={`p-2 rounded hover:bg-border ${showShortcuts ? 'bg-border' : ''}`}
            title="Keyboard Shortcuts"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>

        {/* Save status and word count */}
        <div className="flex items-center gap-4 text-sm text-text-muted">
          {wordCount > 0 && (
            <span className="text-text-secondary">{wordCount} {wordCount === 1 ? 'word' : 'words'}</span>
          )}
          {isSaving ? (
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 border-2 border-text-muted border-t-transparent rounded-full animate-spin" />
              Saving...
            </span>
          ) : lastSaved ? (
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4 text-accent-green" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              Saved at {lastSaved}
            </span>
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
            className="w-full text-xl font-light text-text-primary bg-transparent border-none outline-none focus:ring-0 placeholder-text-muted"
            placeholder="Untitled Note"
          />
        </div>
      )}

      {/* Editor Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin bg-background">
        <EditorContent editor={editor} className="h-full" />
      </div>

      {/* Keyboard Shortcuts Modal */}
      {showShortcuts && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowShortcuts(false)}>
          <div className="bg-background rounded-lg shadow-xl max-w-md w-full mx-4 p-6" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-text-primary">Keyboard Shortcuts</h3>
              <button
                onClick={() => setShowShortcuts(false)}
                className="p-1 hover:bg-border rounded"
              >
                <svg className="w-5 h-5 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium text-text-secondary mb-2">Text Formatting</h4>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm"><span>Bold</span><kbd className="px-2 py-0.5 bg-border rounded text-xs">Ctrl+B</kbd></div>
                  <div className="flex justify-between text-sm"><span>Italic</span><kbd className="px-2 py-0.5 bg-border rounded text-xs">Ctrl+I</kbd></div>
                  <div className="flex justify-between text-sm"><span>Underline</span><kbd className="px-2 py-0.5 bg-border rounded text-xs">Ctrl+U</kbd></div>
                  <div className="flex justify-between text-sm"><span>Heading</span><kbd className="px-2 py-0.5 bg-border rounded text-xs">Alt+H</kbd></div>
                </div>
              </div>
              <div>
                <h4 className="text-sm font-medium text-text-secondary mb-2">Lists</h4>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm"><span>Bullet List</span><kbd className="px-2 py-0.5 bg-border rounded text-xs">- + Space</kbd></div>
                  <div className="flex justify-between text-sm"><span>Numbered List</span><kbd className="px-2 py-0.5 bg-border rounded text-xs">1. + Space</kbd></div>
                </div>
              </div>
              <div>
                <h4 className="text-sm font-medium text-text-secondary mb-2">Editor</h4>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm"><span>Undo</span><kbd className="px-2 py-0.5 bg-border rounded text-xs">Ctrl+Z</kbd></div>
                  <div className="flex justify-between text-sm"><span>Redo</span><kbd className="px-2 py-0.5 bg-border rounded text-xs">Ctrl+Shift+Z</kbd></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
