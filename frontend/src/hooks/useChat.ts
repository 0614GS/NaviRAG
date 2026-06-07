import { useState, useRef, useCallback, useEffect } from 'react';
import { apiChatStream, apiGet } from '../api/client';
import type { UIChatMessage, ChatMessage as ApiChatMessage, SourceRef, MessageSegment } from '../types';

export function useChat(sessionId: string | null) {
  const [messages, setMessages] = useState<UIChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // 切换 session 时加载历史消息
  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      setHistoryLoaded(false);
      return;
    }
    setHistoryLoaded(false);
    apiGet<ApiChatMessage[]>(`/chat/sessions/${sessionId}/messages`)
      .then((apiMessages) => {
        const uiMessages: UIChatMessage[] = apiMessages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          metadata: m.metadata,
        }));
        setMessages(uiMessages);
        setHistoryLoaded(true);
      })
      .catch(() => {
        setMessages([]);
        setHistoryLoaded(true);
      });
  }, [sessionId]);

  const sendMessage = useCallback(
    async (query: string) => {
      if (isLoading || !sessionId) return;

      const userMsg: UIChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: query,
      };

      const assistantId = `assistant-${Date.now()}`;
      const assistantMsg: UIChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        isStreaming: true,
        segments: [],
        sources: [],
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsLoading(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await apiChatStream(
          query,
          sessionId,
          (event) => {
            switch (event.type) {
              case 'token':
                setMessages((prev) =>
                  prev.map((m) => {
                    if (m.id !== assistantId) return m;
                    const segments = [...(m.segments || [])];
                    const last = segments[segments.length - 1];
                    if (last && last.type === 'text') {
                      segments[segments.length - 1] = {
                        ...last,
                        content: last.content + (event.content as string),
                      };
                    } else {
                      segments.push({ type: 'text', content: event.content as string });
                    }
                    return {
                      ...m,
                      content: m.content + (event.content as string),
                      segments,
                    };
                  }),
                );
                break;

              case 'status': {
                const stage = event.stage as string | undefined;
                if (!stage) break; // 忽略无 stage 的通用状态（如 "正在检索文档..."）
                setMessages((prev) =>
                  prev.map((m) => {
                    if (m.id !== assistantId) return m;
                    const segments = [...(m.segments || [])];
                    // 避免重复添加同一个 running 工具
                    const hasRunning = segments.some(
                      (s) => s.type === 'tool_call' && s.name === stage && s.status === 'running',
                    );
                    if (hasRunning) return m;
                    segments.push({
                      type: 'tool_call',
                      name: stage,
                      status: 'running',
                    });
                    return { ...m, segments };
                  }),
                );
                break;
              }

              case 'retrieval': {
                const stage = event.stage as string;
                setMessages((prev) =>
                  prev.map((m) => {
                    if (m.id !== assistantId) return m;
                    const segments = [...(m.segments || [])];
                    // 从后往前找匹配的 running 工具调用
                    for (let i = segments.length - 1; i >= 0; i--) {
                      const s = segments[i];
                      if (s.type === 'tool_call' && s.name === stage && s.status === 'running') {
                        segments[i] = {
                          ...s,
                          status: 'done' as const,
                          content: event.content,
                        };
                        break;
                      }
                    }
                    return { ...m, segments };
                  }),
                );
                break;
              }

              case 'done': {
                const doneContent = event.content as Record<string, unknown> | undefined;
                const sources = (doneContent?.sources as SourceRef[]) || [];
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, isStreaming: false, sources, metadata: doneContent }
                      : m,
                  ),
                );
                setIsLoading(false);
                break;
              }

              case 'error':
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: `错误: ${event.content}`, isStreaming: false }
                      : m,
                  ),
                );
                setIsLoading(false);
                break;
            }
          },
          controller.signal,
        );
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: `请求失败: ${(err as Error).message}`, isStreaming: false }
                : m,
            ),
          );
        }
        setIsLoading(false);
      }
    },
    [isLoading, sessionId],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
    setHistoryLoaded(false);
  }, []);

  return { messages, isLoading, historyLoaded, sendMessage, cancel, clear };
}
