export interface Document {
  doc_id: string;
  doc_name: string;
  summary: string | null;
  keywords: string[];
  status: 'pending' | 'indexing' | 'indexed' | 'failed';
  created_at: string | null;
}

export interface DocumentDetail extends Document {
  filename: string;
  tree_structure: Record<string, unknown> | null;
  error_msg: string | null;
  updated_at: string | null;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  metadata: Record<string, unknown>;
  created_at: string | null;
}

export interface SSEEvent {
  type: 'status' | 'retrieval' | 'token' | 'done' | 'error';
  stage?: string;
  content: unknown;
}

export interface ToolCall {
  name: string;
  status: 'running' | 'done';
  content?: unknown;
}

export interface SourceRef {
  index: number;
  node_id: string;
  doc_id: string;
  title: string;
  path: string;
}

export interface NodeContent {
  node_id: string;
  doc_id: string;
  title: string;
  path: string;
  content: string;
  summary: string;
  keywords: string[];
  level: number;
}

export interface TextSegment {
  type: 'text';
  content: string;
}

export interface ToolCallSegment {
  type: 'tool_call';
  name: string;
  status: 'running' | 'done';
  content?: unknown;
}

export type MessageSegment = TextSegment | ToolCallSegment;

export interface UIChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  isStreaming?: boolean;
  metadata?: Record<string, unknown>;
  toolCalls?: ToolCall[];
  sources?: SourceRef[];
  segments?: MessageSegment[];
}
