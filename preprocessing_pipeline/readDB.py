import json
# from attr import asdict
from graph.graph import GraphDB, Vertex, Edge
from dataclasses import dataclass, asdict
from typing import Dict, List, Union

@dataclass
class Node:
    id: str
    content: str
    type: str
    edges: List[str]
    parent_id: Union[str, None] = None
    
def read_all_Edges_Vertices(db_path, json_file_path):
    g = GraphDB(db_path)
    data = {"vertices": {}, "edges": {}}
    for id, content in g.db.items():
        if id.startswith(Vertex.prefix):
            v = g.get_vertex(id)
            if v is not None:
                data["vertices"][v.id] = asdict(v)
        elif id.startswith(Edge.prefix):
            data["edges"][id] = ""

    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    g.close_db()


def generate_json(db_path, json_file_path):
    graph_db = GraphDB(db_path)
    data_to_export = {}

    # Get all vertex IDs without modifying graphDB source code
    for vertex_id in graph_db.db.keys():
        if vertex_id.startswith(Vertex.prefix):  # Prefix for Vertex IDs
            vertex = graph_db.get_vertex(vertex_id)
            if vertex is not None:
                # Create the Node object with its id, content, and type
                node = Node(id=vertex.id, content=vertex.content, type=vertex.type, edges=[], parent_id=None)
                # Get the edges starting from this vertex
                edges = graph_db.get_edges(vertex_id)
                # Add the IDs of these edges to the node's edge list
                for edge in edges:
                    if 'child' in edge.id:
                        node.edges.append(edge.id.split(Edge.delimiter)[-1])
                data_to_export[vertex_id] = asdict(node)  # Now this should work

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_export, json_file)

    graph_db.close_db()



json_file_path_raw = "preprocessing_pipeline/output_files/db_export_raw.json"
json_file_path_structured = "preprocessing_pipeline/output_files/db_export_structured.json"
db_filepath = 'preprocessing_pipeline/output_files/test_dict'

read_all_Edges_Vertices(db_filepath, json_file_path_raw)
generate_json(db_filepath, json_file_path_structured)