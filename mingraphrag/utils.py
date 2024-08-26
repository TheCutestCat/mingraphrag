import os, json
import pandas as pd
from openai import OpenAI
import html
import numpy as np

import networkx as nx
from typing import Any, cast
from graspologic.utils import largest_connected_component
from PromptsAndClasses.entity_extraction import EntityExtractionResponseFormat,Entity,Relation

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.duckagi.com/v1/")

def openai_wrapper(system_messages,input_messages,response_format):
    system_messages = {"role": "system", "content": f"{system_messages}"}
    input_messages = {"role": "user", "content": f"{input_messages}"}
    
    messages = [system_messages]
    messages.append(input_messages)
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=response_format,
    )
    completion_result = completion.choices[0].message

    if completion_result.parsed:
        result = completion_result.parsed
        return result
    else:
        print(completion_result.refusal)

def get_embedding(texts, model="text-embedding-3-small"):
   if isinstance(texts, list):
       texts = [text.replace("\n", " ") for text in texts]
       embeddings = client.embeddings.create(input=texts, model=model).data
       return [embedding.embedding for embedding in embeddings]
   else:
       text = texts.replace("\n", " ")
       return client.embeddings.create(input=[text], model=model).data[0].embedding

def get_embedding_similarity(embedding1, embedding2):
  """计算两个 embedding 之间的余弦相似度"""
  return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def process_directory(path):
    # 遍历目录中的文件
    path = os.path.join(path, 'input')
    output = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        
        # 检查是否是文件（而不是目录）
        if os.path.isfile(file_path):
            # 打印文件名称
            print(f"Found file: {filename}")
            
            # 检查文件是否是TXT格式
            if filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    output.append(content)
                    print(f"Read TXT file: {filename}")
            else:
                print(f"File {filename} is not a TXT, skipping.")
    
    if not output: 
        raise ValueError('No TXT files found in this folder') 
    return output

def chunk_text(text, chunk_size=600, overlap=100):
    words = text.split()  # 将文本分割成单词列表
    chunks = []  # 用于存储chunk的列表
    start = 0  # 起始位置

    while start < len(words):
        end = start + chunk_size  # 计算chunk的结束位置
        chunk = words[start:end]  # 取出当前的chunk
        chunks.append(" ".join(chunk))  # 将chunk重新组合成字符串并添加到chunks列表
        start = end - overlap  # 为下一个chunk计算新的起始位置

        # 如果剩余单词不足一个chunk，但又大于overlap时，确保不会跳过
        if start + chunk_size > len(words) and start < len(words):
            chunk = words[start:]  # 获取剩余的所有单词
            chunks.append(" ".join(chunk))
            break

    return chunks

def convert_to_networkx(data: EntityExtractionResponseFormat) -> nx.Graph:
    G = nx.Graph()

    # 添加节点（实体）
    for entity in data.entities:
        G.add_node(entity.entity_name, type=entity.entity_type, description=entity.entity_description)

    # 添加边（关系）
    for relation in data.relations:
        G.add_edge(
            relation.source_entity,
            relation.target_entity,
            description=relation.relationship_description,
            strength=relation.relationship_strength
        )
    
    return G

def convert_to_entity_extraction_format(G: nx.Graph) -> EntityExtractionResponseFormat:
    entities = []
    relations = []

    # 提取节点信息
    for node, attrs in G.nodes(data=True):
        entity = Entity(
            entity_name=node,
            entity_type=attrs.get('type', ''),
            entity_description=attrs.get('description', '')
        )
        entities.append(entity)

    # 提取边信息
    for source, target, attrs in G.edges(data=True):
        relation = Relation(
            source_entity=source,
            target_entity=target,
            relationship_description=attrs.get('description', ''),
            relationship_strength=attrs.get('strength', 0)
        )
        relations.append(relation)

    # 创建 EntityExtractionResponseFormat 对象
    data = EntityExtractionResponseFormat(
        entities=entities,
        relations=relations
    )

    return data

def analyze_dataframe(df: pd.DataFrame, max_str: int = 3000) -> dict:
    # 1. 获取列名
    column_names = df.columns.tolist()

    # 2. 计算每一列的平均长度
    avg_lengths = df.applymap(lambda x: len(str(x))).mean().to_dict()

    # 3. 获取dataframe的长度
    df_length = len(df)

    # 4. 创建JSON并计算字符数量
    info = {
        "column_names": column_names,
        "average_lengths": avg_lengths,
        "dataframe_length": df_length
    }
    
    # json.dumps 使用 ensure_ascii=False 来避免 Unicode 编码
    json_str = json.dumps(info, ensure_ascii=False)
    json_length = len(json_str)

    # 5. 计算剩余空间并填充列值
    remaining_space = max_str - json_length
    story_str  = ""
    
    example_rows = df.iloc[0:3]
    for _, row in example_rows.iterrows():
        for col in column_names:
            value = str(row[col])
            value = value[:500]
            column_str = f"{col}: {value},"
            
            if len(story_str) + len(column_str) <= remaining_space:
                story_str += column_str
            else:
                break

    return {
        "json_info": info,
        "json_length": json_length,
        "fitted_columns_str": story_str
    }

def get_stable_connected_components(graph: nx.Graph):
    """
    Return all connected components of the graph, with nodes and edges sorted in a stable way.
    """
    components = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component).copy()
        subgraph = normalize_node_names(subgraph)
        subgraph = _stabilize_graph(subgraph)
        components.append(subgraph)
    
    # Sort components by size (number of nodes) in descending order
    components.sort(key=lambda x: x.number_of_nodes(), reverse=True)
    
    return components

# some graph functions    
def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """Normalize node names."""
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
    return nx.relabel_nodes(graph, node_mapping)
def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
    graph = graph.copy()
    graph = cast(nx.Graph, largest_connected_component(graph))
    graph = normalize_node_names(graph)
    return _stabilize_graph(graph)
def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure an undirected graph with the same relationships will always be read the same way."""
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    sorted_nodes = graph.nodes(data=True)
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))

    # If the graph is undirected, we create the edges in a stable way, so we get the same results
    # for example:
    # A -> B
    # in graph theory is the same as
    # B -> A
    # in an undirected graph
    # however, this can lead to downstream issues because sometimes
    # consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
    # but they base some of their logic on the order of the nodes, so the order ends up being important
    # so we sort the nodes in the edge in a stable way, so that we always get the same order
    if not graph.is_directed():

        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                temp = source
                source = target
                target = temp
            return source, target, edge_data

        edges = [_sort_source_target(edge) for edge in edges]

    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

    fixed_graph.add_edges_from(edges)
    return fixed_graph

