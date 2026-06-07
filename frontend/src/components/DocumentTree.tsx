import { useState } from 'react';
import { ChevronRight, ChevronDown, FileText, Loader2 } from 'lucide-react';
import type { NodeContent } from '../types';
import { apiGet } from '../api/client';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface TreeNode {
  title: string;
  path: string;
  node_id?: string;
  nodes?: TreeNode[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function toTreeNodes(nodes: unknown): TreeNode[] {
  if (!Array.isArray(nodes)) return [];
  return nodes.map((n: Record<string, unknown>) => ({
    title: String(n.title || ''),
    path: String(n.path || ''),
    node_id: n.node_id ? String(n.node_id) : undefined,
    nodes: n.nodes ? toTreeNodes(n.nodes) : undefined,
  }));
}

interface Props {
  data: unknown;
  docName?: string;
}

export default function DocumentTree({ data, docName }: Props) {
  const nodes = toTreeNodes((data as Record<string, unknown>)?.nodes);
  if (nodes.length === 0) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-100 bg-gray-50">
        <h3 className="font-semibold text-gray-800 flex items-center gap-2">
          <FileText size={18} className="text-indigo-500" />
          {docName || '文档结构'}
        </h3>
      </div>
      <div className="p-2">
        {nodes.map((node, i) => (
          <TreeNodeItem key={i} node={node} depth={0} index={i} />
        ))}
      </div>
    </div>
  );
}

function TreeNodeItem({ node, depth, index }: { node: TreeNode; depth: number; index: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const [showContent, setShowContent] = useState(false);
  const [nodeData, setNodeData] = useState<NodeContent | null>(null);
  const [loading, setLoading] = useState(false);
  const hasChildren = node.nodes && node.nodes.length > 0;
  const hasNodeId = !!node.node_id;

  const handleToggleChildren = () => {
    if (hasChildren) setExpanded(!expanded);
  };

  const handleToggleContent = async () => {
    if (!hasNodeId) return;
    if (showContent) {
      setShowContent(false);
      return;
    }
    setLoading(true);
    setShowContent(true);
    try {
      const data = await apiGet<NodeContent>(`/nodes/${node.node_id}`);
      setNodeData(data);
    } catch {
      setNodeData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div
        className="tree-node flex items-center gap-1 px-2 py-1.5 rounded-md text-sm transition-colors"
        style={{ paddingLeft: `${12 + depth * 20}px` }}
      >
        {/* Expand/collapse children */}
        <button
          className="shrink-0 p-0.5"
          onClick={handleToggleChildren}
        >
          {hasChildren ? (
            expanded ? (
              <ChevronDown size={14} className="text-gray-400" />
            ) : (
              <ChevronRight size={14} className="text-gray-400" />
            )
          ) : (
            <span className="w-[18px] shrink-0" />
          )}
        </button>

        {/* Node title - click to show content */}
        <button
          className={`text-left truncate hover:text-indigo-600 transition-colors ${
            hasNodeId ? 'cursor-pointer' : 'cursor-default'
          } ${showContent && nodeData ? 'text-indigo-700 font-medium' : 'text-gray-700'}`}
          onClick={handleToggleContent}
        >
          {node.title}
        </button>

        {hasNodeId && showContent && loading && (
          <Loader2 size={11} className="animate-spin text-gray-400 shrink-0 ml-1" />
        )}
      </div>

      {/* Node content */}
      {showContent && nodeData && (
        <div
          className="ml-6 mr-2 mb-2 p-3 bg-indigo-50/50 rounded-lg border border-indigo-100 text-xs"
          style={{ marginLeft: `${32 + depth * 20}px` }}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-gray-400 text-[10px]">{nodeData.path}</span>
            <button
              className="text-gray-400 hover:text-gray-600"
              onClick={() => setShowContent(false)}
            >
              &times;
            </button>
          </div>
          {nodeData.content ? (
            <div className="prose prose-xs max-w-none text-gray-700 prose-headings:text-gray-800 prose-code:text-pink-600 prose-pre:bg-gray-900 prose-pre:text-gray-100">
              <Markdown remarkPlugins={[remarkGfm]}>
                {nodeData.content}
              </Markdown>
            </div>
          ) : (
            <p className="text-gray-400 italic">(此节点无正文内容)</p>
          )}
        </div>
      )}

      {/* Children */}
      {hasChildren && expanded && (
        <div className="tree-indent" style={{ marginLeft: `${19 + depth * 20}px` }}>
          {node.nodes!.map((child, i) => (
            <TreeNodeItem key={i} node={child} depth={depth + 1} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
