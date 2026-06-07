import { useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { useChat } from '../hooks/useChat';
import { MessageSquare } from 'lucide-react';

interface Props {
  sessionId: string | null;
  onNewChat: () => void;
}

export default function ChatPanel({ sessionId, onNewChat }: Props) {
  const { messages, isLoading, sendMessage } = useChat(sessionId);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = (query: string) => {
    if (!sessionId) {
      onNewChat();
      return;
    }
    sendMessage(query);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <MessageSquare size={48} className="mb-4 text-gray-300" />
            <p className="text-lg font-medium mb-2">欢迎使用 NaviRAG</p>
            <p className="text-sm">
              {sessionId
                ? '输入问题开始检索文档…'
                : '点击左侧「新对话」开始'}
            </p>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}
