import glob
import numpy as np
import rocksdict, sys
from datatypes import *
import random
import json
from typing import Optional

def test_decode_embed(index_file_path: str, database_path: str, embed_id: Optional[str] = None):
    # Load index file
    with open(index_file_path, 'r') as json_file:
        index_list = json.load(json_file)

    # Select a node ID
    if embed_id is None:
        random_node = random.choice(index_list)
        node_id = random_node['node_id']
    else:
        node_id = embed_id

    # Load database
    database = rocksdict.Rdict(database_path)
    node = Node(**database[node_id]['node'])
    print(node.content)

    database.close()


def get_all_edge_ids(database_path: str) -> List[str]:
    database = rocksdict.Rdict(database_path)
    edge_ids = []
    for key, value in database.items():
        if 'edge' in value:
            edge = Edge(**value['edge'])
            edge_ids.append(edge.id)
    database.close()
    return edge_ids

def prefix_search_edge_ids(edge_ids: List[str], prefix: str) -> List[str]:
    return [id for id in edge_ids if id.startswith(prefix)]


# edge_ids = get_all_edge_ids('path_to_your_database')
# prefix_edges = prefix_search_edge_ids(edge_ids, 'uid_parent')

# print(prefix_edges)

db_filepath = 'preprocessing_pipeline/output_files/test_dict'
index_file_path = 'preprocessing_pipeline/output_files/vector_store/embed_index_0.idx.json'
test_decode_embed(index_file_path, db_filepath)

