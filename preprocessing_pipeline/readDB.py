import rocksdict
import numpy as np
import json
from dataclasses import asdict, field
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class Attribute:
    name: str
    value: str
    type: str

@dataclass
class Edge:
    source: str
    target: str
    type: str
    attributes: Dict[str, Attribute]

@dataclass
class Node:
    id: str
    content: str
    type: str
    attributes: Dict[str, Attribute]
    edges: List[Edge]
    parent_id: Union[str, None] = None

def read_from_db(db_path, json_file_path):
    db = rocksdict.Rdict(db_path)
    data_to_export = {}
    for k, v in db.items():
        print(f"Raw output from RocksDB: {v}")
        # print("Attempting to load JSON string")
        node_dict = json.loads(v.decode('utf-8'))['node']  # no 'embed' in the dictionary anymore
        node = Node(**node_dict)
        data_to_export[k] = asdict(node)
        # print(f'Node ID: {k}')
        # print(f'Content: {node.content}')
        # print(f'Attributes: {node.attributes}')
        # print(f'Edges: {node.edges}\n')

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_export, json_file)

    db.close()

# Example usage
db_filepath = 'preprocessing_pipeline/test_dict'
json_file_path = 'preprocessing_pipeline/db_export.json'
read_from_db(db_filepath, json_file_path)
