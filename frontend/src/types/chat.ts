export type ChatRole = 'user' | 'assistant'

export interface ChatSessionSummary {
  session_id: string
  title: string | null
  created_at: string
  updated_at: string
  message_count: number
}

export interface ChatSessionPage {
  items: ChatSessionSummary[]
  total: number
  limit: number
  offset: number
}

export interface ChatCitation {
  id: string
  source_number: number
  chunk_id: string
  document_name: string
  pages: number[]
  section: string | null
  excerpt: string
  document_title: string | null
  filename: string | null
  chapter: string | null
  page_start: number | null
  page_end: number | null
  domain: string | null
  chunk_type: string | null
  is_cited: boolean
  created_at: string
}

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  created_at: string
  insufficient_context: boolean | null
  model_name: string | null
  response_language: string | null
  question_type: string | null
  total_ms: number | null
  citations: ChatCitation[]
}

export interface ChatSessionHistory {
  session_id: string
  title: string | null
  created_at: string
  updated_at: string
  messages: ChatMessage[]
}

export interface SendChatRequest {
  question: string
  session_id?: string
  source_mode?: string
  language_mode?: string
}

export interface SendChatResponse {
  session_id: string
  user_message_id: string
  assistant_message_id: string
  question: string
  answer: string
  insufficient_context: boolean
}

export interface RenameSessionResponse {
  session_id: string
  title: string
  updated_at: string
}