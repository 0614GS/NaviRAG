import time

from langgraph.constants import START
from langgraph.graph import StateGraph

from core.workflow.nodes import *
from core.workflow.states import State

search_workflow = (
    StateGraph(State)
    # 选择文件
    .add_node(select_docs)
    # 选择某一个文件内节点
    .add_node(select_nodes)
    # 评分节点
    .add_node(grade_node_content)

    .add_edge(START, "select_docs")
    # map，send到挑选节点 select_nodes
    .add_edge("select_docs", "select_nodes")
    # reduce 汇聚边 并 send到评价节点grade_node_content
    .add_edge("select_nodes", "grade_node_content")
    # 直接汇聚到END
    .add_edge("grade_node_content", END)
    .compile()
)

if __name__ == "__main__":
    # print(node_content_store.mget(["0001"]))
    # for chunk in search_workflow.stream(
    #         {
    #             "query": input("提问：\n")
    #         },
    #         stream_mode="updates",
    # ):
    #     print(chunk)
    print("开始")
    beg = time.time()
    for chunk in search_workflow.stream(
        {"query": "怎么使用长期记忆？"},
        stream_mode="updates"
    ):
        print(chunk)
        print(time.time() - beg)