from asyncio.log import logger

from langchain_core.tools import tool
from core.workflow.graph import search_workflow


@tool("search_local_docs", description="从本地中获取langchain官方文档的关于问题的相关片段，优先使用此工具获取有关langchain文档的信息")
def search_local_docs(query: str) -> list[str]:
    """
    :param query: 想要查询的相关内容，要足够具体
    :return: 相关的文档片段
    """
    response = search_workflow.invoke({"query": query})
    final_nodes = response.get("final_nodes", [])
    print(len(final_nodes))
    if len(final_nodes) == 0:
        return ["未能找到相关内容，请使用其他搜索工具，或者更改你要查询的问题再次向我查询"]
    else:
        # content = ""
        # for i, node in enumerate(final_nodes):
        #     content += f"来源{i}:\n"
        #     content += f"标题:{node['title']}\n"
        #     content += f"路径:{node['path']}\n"
        #     content += f"内容:{node['content']}\n"
        return final_nodes
