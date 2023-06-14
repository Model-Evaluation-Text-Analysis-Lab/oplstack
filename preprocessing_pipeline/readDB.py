import json
# from attr import asdict
from graph.graph import GraphDB
from dataclasses import dataclass, asdict
from typing import Dict, List, Union

@dataclass
class Edge:
    id: str
    source: str
    target: str
    type: str

@dataclass
class Node:
    id: str
    content: str
    type: str
    edges: List[str]
    parent_id: Union[str, None] = None


def read_from_db(db_path, json_file_path):
    graph_db = GraphDB(db_path)
    data_to_export = {}

    # Get all vertex IDs without modifying graphDB source code
    for vertex_id in graph_db.db.keys():
        if vertex_id.startswith("V_"):  # Prefix for Vertex IDs
            vertex = graph_db.get_vertex(vertex_id)
            if vertex is not None:
                node = Node(id=vertex.id, content=vertex.content, type=vertex.type, edges=[], parent_id=None)
                data_to_export[vertex_id] = asdict(node)  # Now this should work

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_export, json_file)

    graph_db.close_db()
