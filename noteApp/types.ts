/**
 * TypeScript type definitions for the NoteApp
 *
 * Contains all interfaces and types used across the application
 */

// ============================================================================
// User & Authentication Types
// ============================================================================

export interface User {
  id: string
  email: string
  name: string
  avatar_url?: string
  created_at: string
  updated_at: string
}

export interface UserProfile {
  id: string
  user_id: string
  theme_preference: 'light' | 'dark'
  created_at: string
  updated_at: string
}

export interface AuthSession {
  user: User
  access_token: string
  refresh_token: string
  expires_at: number
}

// ============================================================================
// Note Types
// ============================================================================

export interface Note {
  id: string
  user_id: string
  title: string
  content: string // Tiptap JSON or HTML
  plain_text: string // For search
  is_deleted: boolean
  deleted_at?: string
  is_pinned: boolean
  pinned_at?: string
  created_at: string
  updated_at: string
  tags: Tag[]
  jira_issue_key?: string // e.g., "PROJ-123"
  shared_with: NoteShare[]
}

export interface NoteCreate {
  title: string
  content: string
  plain_text: string
  tags?: string[]
}

export interface NoteUpdate {
  title?: string
  content?: string
  plain_text?: string
  tags?: string[]
}

export interface NoteShare {
  id: string
  note_id: string
  user_id: string
  shared_with_email: string
  permission: 'view' | 'edit'
  created_at: string
}

// ============================================================================
// Tag Types
// ============================================================================

export interface Tag {
  id: string
  user_id: string
  name: string
  color: string
  created_at: string
}

// ============================================================================
// Jira Types
// ============================================================================

export interface JiraConfig {
  id: string
  user_id: string
  domain: string // e.g., "yourcompany.atlassian.net"
  email: string
  api_token: string // encrypted
  default_project: string
  default_issue_type: string
  created_at: string
  updated_at: string
}

export interface JiraTask {
  summary: string // Task headline
  description: string // Task description
  acceptanceCriteria: string[] // List of ACs
}

export interface JiraDetectionState {
  detected: boolean
  task: JiraTask | null
  confidence: 'low' | 'medium' | 'high'
}

export interface JiraIssue {
  id: string
  key: string // e.g., "PROJ-123"
  summary: string
  description: string
  status: string
  assignee?: string
  created: string
  updated: string
  url: string
}

export interface JiraCreateRequest {
  project: string
  issueType: string
  summary: string
  description: string
  acceptanceCriteria?: string[]
}

// ============================================================================
// Gmail Types
// ============================================================================

export interface GmailConfig {
  id: string
  user_id: string
  email: string
  access_token: string // encrypted
  refresh_token: string // encrypted
  expires_at: number
  created_at: string
  updated_at: string
}

export interface EmailTemplate {
  id: string
  user_id: string
  name: string
  subject: string
  body: string // Supports variables like {{noteTitle}}, {{noteContent}}
  created_at: string
}

export interface EmailRequest {
  to: string
  subject: string
  body: string
  attachMarkdown?: boolean
  noteId?: string
  templateId?: string
}

export interface EmailResponse {
  success: boolean
  messageId?: string
  error?: string
}

// ============================================================================
// File Sync Types
// ============================================================================

export interface SyncConfig {
  id: string
  user_id: string
  folder_path: string
  last_sync: string
  sync_enabled: boolean
}

export interface SyncStatus {
  status: 'synced' | 'syncing' | 'error' | 'conflict'
  lastSync?: string
  error?: string
}

export interface MarkdownFile {
  path: string
  filename: string
  content: string
  frontmatter?: YAMLFrontmatter
  lastModified: string
}

export interface YAMLFrontmatter {
  title?: string
  tags?: string[]
  date?: string
  [key: string]: any
}

// ============================================================================
// UI State Types
// ============================================================================

export interface PanelState {
  left: boolean // notes list panel
  right: boolean // tags and actions panel
}

export interface EditorState {
  isSaving: boolean
  lastSaved?: string
  hasUnsavedChanges: boolean
}

export interface SearchState {
  query: string
  results: Note[]
  isSearching: boolean
}

// ============================================================================
// API Response Types
// ============================================================================

export interface ApiResponse<T> {
  data?: T
  error?: ApiError
  success: boolean
}

export interface ApiError {
  message: string
  code: string
  details?: any
}

export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

// ============================================================================
// Utility Types
// ============================================================================

export type SortOrder = 'asc' | 'desc'
export type SortBy = 'created_at' | 'updated_at' | 'title'

export interface NotesFilter {
  tags?: string[]
  search?: string
  isDeleted?: boolean
  sortBy?: SortBy
  sortOrder?: SortOrder
  page?: number
  pageSize?: number
}
