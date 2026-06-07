import { useState, useEffect, useCallback } from 'react';
import { FileText, Trash2, RefreshCw, CheckCircle, Clock, AlertCircle, Loader2, ChevronDown } from 'lucide-react';
import DocumentUpload from './DocumentUpload';
import DocumentTree from './DocumentTree';
import { apiGet, apiDelete, apiPost } from '../api/client';
import type { Document, DocumentDetail } from '../types';

export default function DocumentList() {
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [docDetail, setDocDetail] = useState<DocumentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const loadDocs = useCallback(async () => {
    try {
      const data = await apiGet<{ documents: Document[]; total: number }>('/documents?size=100');
      setDocs(data.documents);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  // 轮询索引中的文档
  useEffect(() => {
    const hasPending = docs.some((d) => d.status === 'pending' || d.status === 'indexing');
    if (!hasPending) return;

    const timer = setInterval(loadDocs, 3000);
    return () => clearInterval(timer);
  }, [docs, loadDocs]);

  // 点击展开/收起文档树
  const toggleDocDetail = async (docId: string) => {
    if (expandedDoc === docId) {
      setExpandedDoc(null);
      setDocDetail(null);
      return;
    }
    setExpandedDoc(docId);
    setDetailLoading(true);
    try {
      const detail = await apiGet<DocumentDetail>(`/documents/${docId}`);
      setDocDetail(detail);
    } catch {
      setDocDetail(null);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await apiDelete(`/documents/${docId}`);
      setDocs((prev) => prev.filter((d) => d.doc_id !== docId));
    } catch {
      // ignore
    }
  };

  const handleReindex = async (docId: string) => {
    try {
      await apiPost(`/documents/${docId}/reindex`);
      loadDocs();
    } catch {
      // ignore
    }
  };

  const statusBadge = (status: string) => {
    switch (status) {
      case 'indexed':
        return (
          <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-green-100 text-green-700">
            <CheckCircle size={12} /> 已索引
          </span>
        );
      case 'indexing':
        return (
          <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-blue-100 text-blue-700">
            <Loader2 size={12} className="animate-spin" /> 索引中
          </span>
        );
      case 'pending':
        return (
          <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-yellow-100 text-yellow-700">
            <Clock size={12} /> 等待中
          </span>
        );
      case 'failed':
        return (
          <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-red-100 text-red-700">
            <AlertCircle size={12} /> 失败
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="p-6 max-w-5xl mx-auto w-full">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-800">文档管理</h2>
          <p className="text-sm text-gray-400 mt-1">上传 Markdown 文档并管理索引</p>
        </div>
        <DocumentUpload onUploaded={loadDocs} />
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <Loader2 size={32} className="animate-spin text-gray-400" />
        </div>
      ) : docs.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <FileText size={48} className="mx-auto mb-4 text-gray-300" />
          <p>暂无文档，点击上方按钮上传</p>
        </div>
      ) : (
        <div className="space-y-2">
          {docs.map((doc) => (
            <div key={doc.doc_id} className="bg-white rounded-xl border border-gray-200 overflow-hidden">
              {/* 文档行 */}
              <div
                className="flex items-center px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => toggleDocDetail(doc.doc_id)}
              >
                <ChevronDown
                  size={16}
                  className={`text-gray-400 shrink-0 mr-2 transition-transform ${
                    expandedDoc === doc.doc_id ? '' : '-rotate-90'
                  }`}
                />
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <FileText size={16} className="text-gray-400 shrink-0" />
                  <span className="text-sm font-medium text-gray-800 truncate">{doc.doc_name}</span>
                </div>
                <div className="mr-3">{statusBadge(doc.status)}</div>
                <p className="text-xs text-gray-400 truncate max-w-xs hidden md:block mr-3 flex-1">
                  {doc.summary || '-'}
                </p>
                <div className="flex items-center gap-1 shrink-0" onClick={(e) => e.stopPropagation()}>
                  <button
                    onClick={() => handleReindex(doc.doc_id)}
                    className="p-1.5 rounded-lg hover:bg-blue-50 text-gray-400 hover:text-blue-600 transition-colors"
                    title="重新索引"
                  >
                    <RefreshCw size={14} />
                  </button>
                  <button
                    onClick={() => handleDelete(doc.doc_id)}
                    className="p-1.5 rounded-lg hover:bg-red-50 text-gray-400 hover:text-red-600 transition-colors"
                    title="删除"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>

              {/* 展开的文档树 */}
              {expandedDoc === doc.doc_id && (
                <div className="border-t border-gray-100">
                  {detailLoading ? (
                    <div className="flex justify-center py-8">
                      <Loader2 size={20} className="animate-spin text-gray-400" />
                    </div>
                  ) : docDetail?.tree_structure ? (
                    <DocumentTree
                      data={docDetail.tree_structure}
                      docName={docDetail.doc_name}
                    />
                  ) : (
                    <div className="py-4 text-center text-sm text-gray-400">
                      {doc.status !== 'indexed' ? '索引完成后可查看文档结构' : '暂无可显示的文档结构'}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
