"""Markdown 解析与树构建 — 纯函数，无 LLM 依赖

从原 data/md2tree.py 提取的 extract_nodes_from_markdown() 和 build_tree_from_flat_nodes()
"""

import re
from typing import List, Dict


def extract_nodes_from_markdown(markdown_content: str) -> List[Dict]:
    """
    解析 Markdown 内容，将每个标题及其下方的文本提取为一个节点列表。

    Args:
        markdown_content: Markdown 文件内容字符串

    Returns:
        包含 'level', 'title', 'text' 的扁平节点列表
    """
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'

    lines = markdown_content.split('\n')
    node_list = []
    current_node = None
    in_code_block = False

    for line in lines:
        stripped_line = line.strip()

        # 1. 状态检查：是否在代码块中
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block

        # 2. 检查是否是标题行（且不在代码块中）
        header_match = re.match(header_pattern, stripped_line)
        if header_match and not in_code_block:
            # 保存之前的节点
            if current_node:
                current_node['text'] = '\n'.join(current_node['text_lines']).strip()
                del current_node['text_lines']
                node_list.append(current_node)

            # 创建新节点
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_node = {
                'level': level,
                'title': title,
                'text_lines': []
            }
        else:
            if current_node:
                current_node['text_lines'].append(line)

    # 处理最后一个节点
    if current_node:
        current_node['text'] = '\n'.join(current_node['text_lines']).strip()
        del current_node['text_lines']
        node_list.append(current_node)

    return node_list


def build_tree_from_flat_nodes(node_list: List[Dict]) -> List[Dict]:
    """
    使用栈算法将扁平的节点列表转换为嵌套的树结构。
    """
    if not node_list:
        return []

    stack = []  # [(node, level), ...]
    root_nodes = []

    for node in node_list:
        current_level = node['level']

        tree_node = {
            'title': node['title'],
            'text': node['text'],
            'nodes': []
        }

        # 弹出直到找到父节点
        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, _ = stack[-1]
            parent_node['nodes'].append(tree_node)

        stack.append((tree_node, current_level))

    return root_nodes


def parse_document(content: str) -> List[Dict]:
    """解析文档内容并构建树结构（整合函数）"""
    flat_nodes = extract_nodes_from_markdown(content)
    return build_tree_from_flat_nodes(flat_nodes)
