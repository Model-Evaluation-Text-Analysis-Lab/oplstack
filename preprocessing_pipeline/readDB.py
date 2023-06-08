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
    id: str
    source: str
    target: str
    type: str
    attributes: dict

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
        item_dict = json.loads(v.decode('utf-8'))
        # print(item_dict)
        if 'node' in item_dict:
            node = Node(**item_dict['node'])
            data_to_export[k] = asdict(node)
        elif 'edge' in item_dict:
            edge = Edge(**item_dict['edge'])
            # May need to handle the edge data export to JSON
            # data_to_export[k] = asdict(edge)

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_export, json_file)

    db.close()



# Example usage
db_filepath = 'preprocessing_pipeline/test_dict'
json_file_path = 'preprocessing_pipeline/db_export.json'
read_from_db(db_filepath, json_file_path)
