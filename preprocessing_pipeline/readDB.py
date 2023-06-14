import rocksdict
import json
from dataclasses import asdict
from datatypes import Node, Edge, Attribute

def read_from_db(db_path, json_file_path):
    db = rocksdict.Rdict(db_path)
    data_to_export = {}

    for k, v in db.items():
        if 'node' in v:
            node_data = v['node']
            node_edges = []
            
            # Create Edge objects from edges list
            if 'edges' in node_data:
                for edge_id in node_data['edges']:
                    edge_data = db[edge_id].get('edge', {})
                    
                    # Create Attribute objects from edge attributes
                    attributes = {attr_name: Attribute(name=attr_name, value=attr_value, type='str') for attr_name, attr_value in edge_data.get('attributes', {}).items()}
                    
                    # Create Edge object
                    edge = Edge(id=edge_id, source=edge_data.get('source', ''), target=edge_data.get('target', ''), type=edge_data.get('type', ''), attributes=attributes)
                    node_edges.append(edge)
            
            # Create Attribute objects from node attributes
            node_attributes = {attr_name: Attribute(name=attr_name, value=attr_value, type='str') for attr_name, attr_value in node_data.get('attributes', {}).items()}
            
            # Create Node object
            node = Node(id=k, content=node_data.get('content', ''), type=node_data.get('type', ''), attributes=node_attributes, edges=node_edges, parent_id=node_data.get('parent_id'))
            
            data_to_export[k] = asdict(node)

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_export, json_file)

    db.close()

# Example usage
db_filepath = 'preprocessing_pipeline/output_files/test_dict'
json_file_path = 'preprocessing_pipeline/output_files/db_export.json'
read_from_db(db_filepath, json_file_path)
