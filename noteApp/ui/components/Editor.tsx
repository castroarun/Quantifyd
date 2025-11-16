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
import { useEffect, useState, useCallback } from 'react'
import { createClient } from '@/lib/supabase-client'
import type { Note } from '@/types'
import { JiraDetectionPanel } from './JiraDetectionPanel'

interface EditorProps {
  noteId: string | null
  userId: string
}

export function Editor({ noteId, userId }: EditorProps) {
  const [note, setNote] = useState<Note | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const [lastSaved, setLastSaved] = useState<string | null>(null)
  const [saveTimeout, setSaveTimeout] = useState<NodeJS.Timeout | null>(null)
  const supabase = createClient()

  /**
   * Initialize Tiptap editor with extensions
   */
  const editor = useEditor({
    immediatelyRender: false,
    extensions: [
      StarterKit,
      Link.configure({
        openOnClick: false,
      }),
      Placeholder.configure({
        placeholder: 'Start typing your note...',
      }),
    ],
    content: '',
    editorProps: {
      attributes: {
        class: 'prose prose-sm sm:prose lg:prose-lg xl:prose-xl focus:outline-none min-h-full p-8',
      },
    },
    onUpdate: ({ editor }) => {
      // Trigger auto-save on content change
      handleAutoSave(editor.getHTML(), editor.getText())
    },
  })

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
   * Calls: supabase.from('notes').upsert
   */
  const saveNote = async (content: string, plainText: string) => {
    setIsSaving(true)

    try {
      const noteData = {
        id: noteId || undefined,
        user_id: userId,
        title: plainText.substring(0, 100) || 'Untitled',
        content,
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
      editor?.commands.setContent(data.content)
    }

    loadNote()
  }, [noteId, editor, supabase])

  if (!editor) {
    return <div className="flex items-center justify-center h-full">Loading editor...</div>
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Editor Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-background">
        <button
          onClick={() => editor.chain().focus().toggleBold().run()}
          className={`p-2 rounded hover:bg-border-light transition-colors ${
            editor.isActive('bold') ? 'bg-border-light' : ''
          }`}
          title="Bold (Ctrl+B)"
        >
          <strong className="text-text-primary">B</strong>
        </button>
        <button
          onClick={() => editor.chain().focus().toggleItalic().run()}
          className={`p-2 rounded hover:bg-border-light transition-colors ${
            editor.isActive('italic') ? 'bg-border-light' : ''
          }`}
          title="Italic (Ctrl+I)"
        >
          <em className="text-text-primary">I</em>
        </button>
        <button
          onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
          className={`p-2 rounded hover:bg-border-light transition-colors ${
            editor.isActive('heading', { level: 1 }) ? 'bg-border-light' : ''
          }`}
          title="Heading 1"
        >
          <span className="text-text-primary font-bold">H1</span>
        </button>
        <button
          onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
          className={`p-2 rounded hover:bg-border-light transition-colors ${
            editor.isActive('heading', { level: 2 }) ? 'bg-border-light' : ''
          }`}
          title="Heading 2"
        >
          <span className="text-text-primary font-bold">H2</span>
        </button>
        <button
          onClick={() => editor.chain().focus().toggleBulletList().run()}
          className={`p-2 rounded hover:bg-border-light transition-colors ${
            editor.isActive('bulletList') ? 'bg-border-light' : ''
          }`}
          title="Bullet List"
        >
          <span className="text-text-primary">•</span>
        </button>
        <button
          onClick={() => editor.chain().focus().toggleOrderedList().run()}
          className={`p-2 rounded hover:bg-border-light transition-colors ${
            editor.isActive('orderedList') ? 'bg-border-light' : ''
          }`}
          title="Numbered List"
        >
          <span className="text-text-primary">1.</span>
        </button>
        <button
          onClick={() => editor.chain().focus().toggleCodeBlock().run()}
          className={`p-2 rounded hover:bg-border-light transition-colors ${
            editor.isActive('codeBlock') ? 'bg-border-light' : ''
          }`}
          title="Code Block"
        >
          <span className="text-text-primary font-mono">{'<>'}</span>
        </button>

        {/* Save status indicator */}
        <div className="ml-auto flex items-center gap-2 text-sm text-text-muted">
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

      {/* Editor Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin bg-background">
        <EditorContent editor={editor} className="h-full" />
      </div>

      {/* Jira Detection Panel (appears when Jira task is detected) */}
      {editor && <JiraDetectionPanel editor={editor} noteId={noteId} />}
    </div>
  )
}
