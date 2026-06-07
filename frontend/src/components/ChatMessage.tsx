import { useState } from 'react';
import { User, Bot, Wrench, ChevronDown, ChevronRight, FileText, ExternalLink, Loader2 } from 'lucide-react';
import type { UIChatMessage, SourceRef, NodeContent, MessageSegment, ToolCallSegment } from '../types';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiGet } from '../api/client';

export default function ChatMessage({ message }: { message: UIChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="flex gap-3 mb-6 justify-end">
        <div className="max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed bg-indigo-600 text-white rounded-br-md">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
        <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center shrink-0">
          <User size={18} className="text-gray-600" />
        </div>
      </div>
    );
  }

  if (message.role === 'system') {
    return (
      <div className="flex justify-center my-2">
        <div className="inline-flex items-start gap-2 px-3 py-1.5 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-700 max-w-[80%]">
          <Wrench size={12} className="shrink-0 mt-0.5" />
          <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
        </div>
      </div>
    );
  }

  // Assistant message
  const hasSegments = message.segments && message.segments.length > 0;

  return (
    <div className="flex gap-3 mb-6">
      <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center shrink-0">
        <Bot size={18} className="text-indigo-600" />
      </div>

      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed bg-white border border-gray-200 text-gray-800 rounded-bl-md shadow-sm
          ${message.isStreaming ? 'streaming-cursor' : ''}`}
      >
        {/* Segments mode: interspersed text and tool calls */}
        {hasSegments ? (
          <SegmentsRenderer segments={message.segments!} />
        ) : message.content ? (
          <div className="prose prose-sm max-w-none prose-headings:text-gray-900 prose-a:text-indigo-600 prose-code:text-pink-600 prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-table:border-collapse">
            <Markdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </Markdown>
          </div>
        ) : (
          <p className="text-gray-400 italic">思考中...</p>
        )}

        {/* Legacy tool calls (for historical messages without segments) */}
        {!hasSegments && message.toolCalls && message.toolCalls.length > 0 && (
          <ToolCallsSection toolCalls={message.toolCalls} />
        )}

        {/* Source references */}
        {message.sources && message.sources.length > 0 && (
          <SourcesSection sources={message.sources} />
        )}
      </div>
    </div>
  );
}

function SegmentsRenderer({ segments }: { segments: MessageSegment[] }) {
  return (
    <div>
      {segments.map((seg, i) => {
        if (seg.type === 'text') {
          if (!seg.content) return null;
          return (
            <div
              key={i}
              className="prose prose-sm max-w-none prose-headings:text-gray-900 prose-a:text-indigo-600 prose-code:text-pink-600 prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-table:border-collapse"
            >
              <Markdown remarkPlugins={[remarkGfm]}>
                {seg.content}
              </Markdown>
            </div>
          );
        }
        // tool_call segment
        return <InlineToolCall key={i} tc={seg} />;
      })}
    </div>
  );
}

function InlineToolCall({ tc }: { tc: ToolCallSegment }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="my-2 text-xs">
      <button
        className="flex items-center gap-1.5 w-full text-left px-2 py-1 rounded-md bg-gray-50 border border-gray-100 hover:bg-gray-100 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {tc.status === 'running' ? (
          <Loader2 size={11} className="animate-spin text-blue-500 shrink-0" />
        ) : (
          <span className="text-green-500 shrink-0" style={{ width: 11, textAlign: 'center' }}>
            &#10003;
          </span>
        )}
        <Wrench size={11} className="text-gray-400 shrink-0" />
        <span className="text-gray-600">{formatToolName(tc.name)}</span>
        {tc.status === 'done' && (
          <span className="text-gray-400">· {formatToolSummary(tc.name, tc.content)}</span>
        )}
        {tc.status === 'running' && (
          <span className="text-gray-400">检索中...</span>
        )}
        {tc.status === 'done' && (
          expanded ? (
            <ChevronDown size={10} className="text-gray-400 ml-auto" />
          ) : (
            <ChevronRight size={10} className="text-gray-400 ml-auto" />
          )
        )}
      </button>

      {expanded && tc.status === 'done' && (
        <div className="mt-1 px-2 py-1.5 bg-gray-50 rounded border border-gray-100 text-gray-600 whitespace-pre-wrap leading-relaxed">
          {formatToolDetail(tc.name, tc.content)}
        </div>
      )}
    </div>
  );
}

function formatToolName(name: string): string {
  switch (name) {
    case 'list_documents':
      return '列出文档';
    case 'get_doc_tree':
      return '查看文档结构';
    case 'get_node_content':
      return '读取节点内容';
    default:
      return name;
  }
}

function formatToolSummary(name: string, content: unknown): string {
  const data = content as Record<string, unknown> | undefined;
  if (!data) return '完成';

  switch (name) {
    case 'list_documents': {
      const count = data.count as number;
      return count ? `找到 ${count} 个文档` : '暂无可用文档';
    }
    case 'get_doc_tree': {
      const docName = data.doc_name || '';
      const nodeCount = data.node_count || 0;
      return `${docName} · ${nodeCount} 个节点`;
    }
    case 'get_node_content': {
      const count = data.count || 0;
      return `读取了 ${count} 个节点`;
    }
    default:
      return '完成';
  }
}

function formatToolDetail(name: string, content: unknown): string {
  const data = content as Record<string, unknown> | undefined;
  if (!data) return '';

  switch (name) {
    case 'list_documents': {
      const docs = data.preview as Array<Record<string, string>> | undefined;
      if (!docs || docs.length === 0) return '暂无可用文档';
      return docs.map((d) => `- **${d.doc_name}** (${d.doc_id}): ${d.summary}`).join('\n');
    }
    case 'get_doc_tree': {
      const docName = data.doc_name || '';
      const nodeCount = data.node_count || 0;
      const summary = data.summary || '';
      return `**文档**: ${docName}\n**节点数**: ${nodeCount}\n**摘要**: ${summary}`;
    }
    case 'get_node_content': {
      const count = data.count || 0;
      return `共读取 ${count} 个节点`;
    }
    default:
      return JSON.stringify(content, null, 2).slice(0, 500);
  }
}

function ToolCallsSection({ toolCalls }: { toolCalls: { name: string; status: 'running' | 'done'; content?: unknown }[] }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="mt-3 pt-3 border-t border-gray-100">
      <button
        className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <Wrench size={11} />
        <span>检索过程</span>
        <span className="text-gray-400">({toolCalls.length})</span>
      </button>

      {expanded && (
        <div className="mt-2 space-y-1.5">
          {toolCalls.map((tc, i) => (
            <ToolCallItem key={i} toolCall={tc} />
          ))}
        </div>
      )}
    </div>
  );
}

function ToolCallItem({ toolCall }: { toolCall: { name: string; status: 'running' | 'done'; content?: unknown } }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="text-xs">
      <button
        className="flex items-center gap-1.5 w-full text-left px-2 py-1 rounded hover:bg-gray-50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {toolCall.status === 'running' ? (
          <Loader2 size={11} className="animate-spin text-blue-500 shrink-0" />
        ) : (
          <span className="w-[11px] shrink-0 text-green-500">&#10003;</span>
        )}
        <span className="text-gray-600">{formatToolName(toolCall.name)}</span>
        {toolCall.status === 'done' && (
          <span className="text-gray-400">· {formatToolSummary(toolCall.name, toolCall.content)}</span>
        )}
        {toolCall.status === 'done' && (
          expanded
            ? <ChevronDown size={10} className="text-gray-400 ml-auto" />
            : <ChevronRight size={10} className="text-gray-400 ml-auto" />
        )}
      </button>

      {expanded && toolCall.status === 'done' && (
        <div className="ml-5 mt-1 px-2 py-1.5 bg-gray-50 rounded text-gray-600 whitespace-pre-wrap leading-relaxed border border-gray-100">
          {formatToolDetail(toolCall.name, toolCall.content)}
        </div>
      )}
    </div>
  );
}

function SourcesSection({ sources }: { sources: SourceRef[] }) {
  const [selectedNode, setSelectedNode] = useState<NodeContent | null>(null);
  const [loadingNode, setLoadingNode] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const handleSourceClick = async (nodeId: string, index: number) => {
    if (activeIndex === index && selectedNode) {
      setSelectedNode(null);
      setActiveIndex(null);
      return;
    }
    setActiveIndex(index);
    setLoadingNode(true);
    setSelectedNode(null);
    try {
      const node = await apiGet<NodeContent>(`/nodes/${nodeId}`);
      setSelectedNode(node);
    } catch {
      setSelectedNode(null);
    } finally {
      setLoadingNode(false);
    }
  };

  return (
    <div className="mt-3 pt-3 border-t border-gray-100">
      <div className="flex items-center gap-2 mb-2">
        <FileText size={12} className="text-gray-400" />
        <span className="text-xs text-gray-500 font-medium">引用来源</span>
      </div>

      <div className="space-y-1">
        {sources.map((src) => (
          <div key={src.index}>
            <button
              className={`text-xs text-left inline-flex items-center gap-1 hover:underline transition-colors ${
                activeIndex === src.index ? 'text-indigo-800 font-medium' : 'text-indigo-600 hover:text-indigo-800'
              }`}
              onClick={() => handleSourceClick(src.node_id, src.index)}
            >
              <span className="font-medium">[{src.index}]</span>
              <span>{src.path}</span>
              <ExternalLink size={10} />
            </button>

            {loadingNode && activeIndex === src.index && (
              <div className="mt-1 ml-4 flex items-center gap-1 text-xs text-gray-400">
                <Loader2 size={10} className="animate-spin" /> 加载中...
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Source content popup */}
      {selectedNode && activeIndex !== null && (
        <div className="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-200 text-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium text-gray-700">{selectedNode.title}</span>
            <button
              className="text-gray-400 hover:text-gray-600"
              onClick={() => { setSelectedNode(null); setActiveIndex(null); }}
            >
              &times;
            </button>
          </div>
          <div className="text-gray-500 mb-1">{selectedNode.path}</div>
          {selectedNode.content ? (
            <div className="prose prose-xs max-w-none text-gray-700 mt-2 prose-headings:text-gray-800 prose-code:text-pink-600 prose-pre:bg-gray-900 prose-pre:text-gray-100">
              <Markdown remarkPlugins={[remarkGfm]}>
                {selectedNode.content}
              </Markdown>
            </div>
          ) : (
            <p className="text-gray-400 italic">{selectedNode.summary || '(无内容)'}</p>
          )}
        </div>
      )}
    </div>
  );
}
