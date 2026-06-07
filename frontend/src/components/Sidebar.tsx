import { Plus, MessageSquare, Trash2, FileText } from 'lucide-react';
import type { ChatSession } from '../types';

interface Props {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onViewDocs: () => void;
  activeView: 'chat' | 'docs';
}

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onViewDocs,
  activeView,
}: Props) {
  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <h1 className="text-lg font-bold text-indigo-600">NaviRAG</h1>
        <p className="text-xs text-gray-400 mt-1">智能文档检索助手</p>
      </div>

      {/* Actions */}
      <div className="p-3 space-y-1">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg
                     bg-indigo-50 text-indigo-700 hover:bg-indigo-100 transition-colors"
        >
          <Plus size={16} />
          新对话
        </button>
        <button
          onClick={onViewDocs}
          className={`w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors
            ${activeView === 'docs'
              ? 'bg-gray-100 text-gray-900'
              : 'text-gray-600 hover:bg-gray-50'
            }`}
        >
          <FileText size={16} />
          文档管理
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto p-2">
        <p className="px-3 py-2 text-xs text-gray-400 font-medium">最近对话</p>
        {sessions.length === 0 ? (
          <p className="px-3 py-4 text-sm text-gray-400 text-center">
            暂无对话，点击上方开始
          </p>
        ) : (
          sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer
                text-sm transition-colors
                ${session.id === activeSessionId && activeView === 'chat'
                  ? 'bg-indigo-50 text-indigo-700'
                  : 'text-gray-700 hover:bg-gray-50'
                }`}
            >
              <MessageSquare size={14} className="shrink-0" />
              <span className="flex-1 truncate">{session.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(session.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-all"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
