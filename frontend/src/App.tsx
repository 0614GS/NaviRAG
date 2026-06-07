import { useState, useCallback, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import DocumentList from './components/DocumentList';
import { apiGet, apiPost } from './api/client';
import type { ChatSession } from './types';

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [view, setView] = useState<'chat' | 'docs'>('chat');

  const loadSessions = useCallback(async () => {
    try {
      const data = await apiGet<ChatSession[]>('/chat/sessions');
      setSessions(data);
    } catch {
      // 后端可能还没启动
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleNewChat = useCallback(async () => {
    try {
      const session = await apiPost<ChatSession>('/chat/sessions', { title: '新对话' });
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setView('chat');
    } catch {
      // ignore
    }
  }, []);

  const handleDeleteSession = useCallback(async (id: string) => {
    try {
      await fetch(`/api/v1/chat/sessions/${id}`, { method: 'DELETE' });
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (activeSessionId === id) {
        setActiveSessionId(null);
      }
    } catch {
      // ignore
    }
  }, [activeSessionId]);

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(id) => { setActiveSessionId(id); setView('chat'); }}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onViewDocs={() => setView('docs')}
        activeView={view}
      />
      <main className="flex-1 flex flex-col overflow-hidden">
        {view === 'chat' ? (
          <ChatPanel sessionId={activeSessionId} onNewChat={handleNewChat} />
        ) : (
          <div className="overflow-y-auto h-full">
            <DocumentList />
          </div>
        )}
      </main>
    </div>
  );
}
