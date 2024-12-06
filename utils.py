import os
from typing import Literal
from langchain_openai import ChatOpenAI
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import base64
from io import BytesIO


PLATFORMS = ["ollama"] # ["ollama", "xinference", "fastchat", "openai"]


def get_llm_models(platform_type: Literal[tuple(PLATFORMS)]):
    if platform_type == "ollama":
        import ollama
        llm_models = [model["model"] for model in ollama.list()["models"] if "bert" not in model.details.families]
        return llm_models
    elif platform_type == "xinference":
        from xinference_client import Client
        client = Client()
        llm_models = client.list_models()
        return llm_models

def get_embedding_models(platform_type: Literal[tuple(PLATFORMS)]):
    if platform_type == "ollama":
        import ollama
        embedding_models = [model["model"] for model in ollama.list()["models"] if "bert" in model.details.families]
        return embedding_models
    elif platform_type == "xinference":
        from xinference_client import Client
        client = Client()
        embedding_models = client.list_models()
        return embedding_models


def get_chatllm(
        platform_type: Literal[tuple(PLATFORMS)],
        model: str,
        temperature: float = 0.9
):
    if platform_type == "ollama":
        # from langchain_ollama import ChatOllama
        # return ChatOllama
        return ChatOpenAI(
            temperature=temperature,
            model_name=model,
            streaming=True,
            base_url="http://127.0.0.1:11434/v1",
            api_key="EMPTY",
        )
    elif platform_type == "xinference":
        from langchain_community.llms import Xinference
        return Xinference


def show_graph(graph):
    flow_state = StreamlitFlowState(
                       nodes=[StreamlitFlowNode(
                           id=node.id,
                           pos=(0,0),
                           data={"content": node.id},
                           node_type="input" if node.id == "__start__"
                                             else "output" if node.id == "__end__"
                                             else "default",
                       ) for node in graph.nodes.values()],
                       edges=[StreamlitFlowEdge(
                           id=str(enum),
                           source=edge.source,
                           target=edge.target,
                           animated=True,
                       ) for enum, edge in enumerate(graph.edges)],
                   )
    streamlit_flow('example_flow',
                   flow_state,
                   layout=TreeLayout(direction='down'), fit_view=True
    )


def get_kb_names():
    kb_root = os.path.join(os.path.dirname(__file__), "kb")
    kb_names = [f for f in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, f))]
    return kb_names

def get_embedding_model(
        platform_type: Literal[tuple(PLATFORMS)] = "ollama",
        model: str = "quentinz/bge-large-zh-v1.5:latest",
):
    if platform_type == "ollama":
        # from langchain_ollama import ChatOllama
        # return ChatOllama
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    elif platform_type == "xinference":
        from langchain_community.llms import Xinference
        return Xinference

def get_img_base64(file_name: str) -> str:
    """
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    """
    image_path = os.path.join(os.path.dirname(__file__), "img", file_name)
    # 读取图片
    with open(image_path, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{base_str}"

if __name__ == "__main__":
    get_img_base64("chatchat_avatar.png")