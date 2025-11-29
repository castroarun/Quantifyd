/**
 * Database types generated from Supabase schema
 *
 * To generate these types from your Supabase project:
 * 1. Install Supabase CLI: npm install -g supabase
 * 2. Run: npx supabase gen types typescript --project-id YOUR_PROJECT_ID > lib/database.types.ts
 *
 * For now, we use a generic Database type
 */

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string
          email: string
          name: string | null
          avatar_url: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id: string
          email: string
          name?: string | null
          avatar_url?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          email?: string
          name?: string | null
          avatar_url?: string | null
          created_at?: string
          updated_at?: string
        }
      }
      notes: {
        Row: {
          id: string
          user_id: string
          title: string
          content: string
          plain_text: string
          is_deleted: boolean
          deleted_at: string | null
          is_pinned: boolean
          pinned_at: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          title?: string
          content?: string
          plain_text?: string
          is_deleted?: boolean
          deleted_at?: string | null
          is_pinned?: boolean
          pinned_at?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          title?: string
          content?: string
          plain_text?: string
          is_deleted?: boolean
          deleted_at?: string | null
          is_pinned?: boolean
          pinned_at?: string | null
          created_at?: string
          updated_at?: string
        }
      }
      tags: {
        Row: {
          id: string
          user_id: string
          name: string
          color: string
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          color?: string
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          color?: string
          created_at?: string
        }
      }
      note_tags: {
        Row: {
          note_id: string
          tag_id: string
          created_at: string
        }
        Insert: {
          note_id: string
          tag_id: string
          created_at?: string
        }
        Update: {
          note_id?: string
          tag_id?: string
          created_at?: string
        }
      }
      note_shares: {
        Row: {
          id: string
          note_id: string
          owner_id: string
          shared_with_email: string
          shared_with_id: string | null
          permission: string
          created_at: string
        }
        Insert: {
          id?: string
          note_id: string
          owner_id: string
          shared_with_email: string
          shared_with_id?: string | null
          permission?: string
          created_at?: string
        }
        Update: {
          id?: string
          note_id?: string
          owner_id?: string
          shared_with_email?: string
          shared_with_id?: string | null
          permission?: string
          created_at?: string
        }
      }
      jira_configs: {
        Row: {
          id: string
          user_id: string
          domain: string
          email: string
          api_token_encrypted: string
          default_project: string | null
          default_issue_type: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          domain: string
          email: string
          api_token_encrypted: string
          default_project?: string | null
          default_issue_type?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          domain?: string
          email?: string
          api_token_encrypted?: string
          default_project?: string | null
          default_issue_type?: string | null
          created_at?: string
          updated_at?: string
        }
      }
      sync_configs: {
        Row: {
          id: string
          user_id: string
          folder_path: string
          last_sync: string | null
          sync_enabled: boolean
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          folder_path: string
          last_sync?: string | null
          sync_enabled?: boolean
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          folder_path?: string
          last_sync?: string | null
          sync_enabled?: boolean
          created_at?: string
          updated_at?: string
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
  }
}
