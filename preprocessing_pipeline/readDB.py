from graph.graph import Vertex, Edge, GraphDB
from typing import Dict, Union
import json

def parse_vertex(vertex: Vertex):
    if vertex.content:
        try:
            attributes = json.loads(vertex.content)
        except json.JSONDecodeError:
            print(f"Failed to parse content for vertex {vertex.id}. Content: {vertex.content}")
            attributes = {}
    else:
        attributes = {}
        
    return {
        'id': vertex.id,
        'content': vertex.content,
        'type': vertex.type,
        'attributes': attributes,
        'edges': [],  # Initialize edges as an empty list
        'parent_id': None,  # Initialize parent_id as None
    }

def parse_edge_key(key: str):
    parts = key.split(Edge.delimiter)
    src_id = parts[0].replace(Edge.prefix, '')
    type = parts[1]
    dst_id = parts[2]
    return src_id, type, dst_id

def readDB(graph_db: GraphDB):
    output = {"vertices": {}}

    # Get vertex and edge keys
    vertex_keys = [key for key in graph_db.db.keys() if key.startswith(Vertex.prefix)]
    edge_keys = [key for key in graph_db.db.keys() if key.startswith(Edge.prefix)]

    # Process vertices
    for key in vertex_keys:
        vertex = graph_db.get_vertex(key)
        vertex_dict = parse_vertex(vertex)
        output['vertices'][key] = vertex_dict

    # Process edges
    for key in edge_keys:
        src_id, type, dst_id = parse_edge_key(key)

        edge = {'id': key, 'source': src_id, 'target': dst_id, 'type': type, 'attributes': {}}

        src_vertex = output['vertices'].get(src_id)
        dst_vertex = output['vertices'].get(dst_id)

        # Add edge to the edges list of the source vertex
        if src_vertex:
            src_vertex['edges'].append(edge)

        # Set parent_id of the destination vertex
        if dst_vertex and dst_vertex['parent_id'] is None and src_vertex:
            dst_vertex['parent_id'] = src_vertex['id']

    return output

def write_to_file(output, json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    db_filepath = 'preprocessing_pipeline/output_files/test_dict'
    json_file_path = "preprocessing_pipeline/output_files/db_export.json"

    graph_db = GraphDB(db_filepath)
    output = readDB(graph_db)
    write_to_file(output, json_file_path)
    graph_db.close_db()
