'use client'

/**
 * Jira task detection panel component
 *
 * Input: editor instance, noteId
 * Output: Inline panel showing detected Jira task with create button
 *
 * Called by: Editor component
 * Calls: Jira task detection logic, API endpoint to create Jira issue
 */

import { useEffect, useState } from 'react'
import type { Editor } from '@tiptap/react'
import type { JiraTask, JiraDetectionState } from '@/types'

interface JiraDetectionPanelProps {
  editor: Editor
  noteId: string | null
}

export function JiraDetectionPanel({ editor, noteId }: JiraDetectionPanelProps) {
  const [detectionState, setDetectionState] = useState<JiraDetectionState>({
    detected: false,
    task: null,
    confidence: 'low',
  })
  const [isCreating, setIsCreating] = useState(false)

  /**
   * Detect Jira task patterns in editor content
   *
   * Input: Editor plain text content
   * Output: Updates detection state with parsed task
   *
   * Called by: useEffect on editor content change
   * Calls: parseJiraTask
   */
  useEffect(() => {
    if (!editor) return

    const content = editor.getText()
    const task = parseJiraTask(content)

    if (task) {
      setDetectionState({
        detected: true,
        task,
        confidence: 'high',
      })
    } else {
      setDetectionState({
        detected: false,
        task: null,
        confidence: 'low',
      })
    }
  }, [editor?.state.doc])

  /**
   * Parse Jira task from text content
   *
   * Input: Text content
   * Output: JiraTask object or null
   *
   * Called by: Detection logic
   * Calls: None
   */
  const parseJiraTask = (content: string): JiraTask | null => {
    const lines = content.split('\n').filter(line => line.trim())

    if (lines.length < 3) return null // Need at least 3 lines for a task

    // ONLY detect with explicit keywords - be very strict
    const hasExplicitKeywords = content.match(/TASK:|JIRA:|DESC:|AC:/i)
    if (!hasExplicitKeywords) {
      return null // Don't detect without explicit keywords
    }

    const summary = content.match(/(?:TASK:|JIRA:)\s*(.+)/i)?.[1]?.trim() || ''
    const description = content.match(/DESC:\s*(.+)/i)?.[1]?.trim() || ''
    const acMatch = content.match(/AC:\s*(.+)/i)
    const acceptanceCriteria = acMatch ? [acMatch[1].trim()] : []

    if (summary && description) {
      return {
        summary,
        description,
        acceptanceCriteria,
      }
    }

    return null
  }

  /**
   * Create Jira issue from detected task
   *
   * Input: None (uses detection state)
   * Output: Creates Jira issue via API
   *
   * Called by: Create button click
   * Calls: /api/jira/create endpoint
   */
  const handleCreateJiraIssue = async () => {
    if (!detectionState.task) return

    setIsCreating(true)
    try {
      const response = await fetch('/api/jira/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          noteId,
          task: detectionState.task,
        }),
      })

      const data = await response.json()

      if (!response.ok) throw new Error(data.error || 'Failed to create Jira issue')

      // Show success
      alert(`Jira issue created: ${data.issue.key}`)

      // Clear detection
      setDetectionState({
        detected: false,
        task: null,
        confidence: 'low',
      })
    } catch (error) {
      console.error('Error creating Jira issue:', error)
      alert('Failed to create Jira issue. Please check your Jira configuration.')
    } finally {
      setIsCreating(false)
    }
  }

  if (!detectionState.detected || !detectionState.task) {
    return null
  }

  return (
    <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 w-full max-w-2xl px-4 z-20">
      <div className="bg-accent-green/10 border-2 border-accent-green rounded-lg p-4 shadow-lg animate-pulse-green">
        <div className="flex items-start gap-3">
          <svg className="w-6 h-6 text-accent-green flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>

          <div className="flex-1">
            <h4 className="font-medium text-text-primary mb-2">Jira Task Detected</h4>
            <div className="space-y-2 text-sm">
              <div>
                <span className="font-medium text-text-secondary">Summary:</span>
                <p className="text-text-primary mt-1">{detectionState.task.summary}</p>
              </div>
              {detectionState.task.description && (
                <div>
                  <span className="font-medium text-text-secondary">Description:</span>
                  <p className="text-text-primary mt-1">{detectionState.task.description}</p>
                </div>
              )}
              {detectionState.task.acceptanceCriteria.length > 0 && (
                <div>
                  <span className="font-medium text-text-secondary">Acceptance Criteria:</span>
                  <ul className="list-disc list-inside mt-1 text-text-primary">
                    {detectionState.task.acceptanceCriteria.map((ac, index) => (
                      <li key={index}>{ac}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <button
              onClick={handleCreateJiraIssue}
              disabled={isCreating}
              className="px-4 py-2 bg-accent-green text-white rounded hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium whitespace-nowrap"
            >
              {isCreating ? 'Creating...' : 'Create in Jira'}
            </button>
            <button
              onClick={() => setDetectionState({ detected: false, task: null, confidence: 'low' })}
              className="px-4 py-2 border border-border rounded hover:bg-border-light transition-colors text-sm text-text-secondary whitespace-nowrap"
            >
              Dismiss
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
