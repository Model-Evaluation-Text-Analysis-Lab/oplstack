import rocksdict
import numpy as np
import json
from dataclasses import asdict
from datatypes import Node, Edge, Attribute

def read_from_db(db_path, json_file_path):
    db = rocksdict.Rdict(db_path)
    data_to_export = {}
    edge_dict = {}  # to temporarily hold edges with their IDs
    non_existing_edge_ids = []  # list to store non-existing edge IDs
    valid_edge_count = 0  # to store the count of valid edges

    for k, v in db.items():
        # print(f"Raw output from RocksDB: {v}")
        item_dict = json.loads(v.decode('utf-8'))
        if 'node' in item_dict:
            # Transform edge IDs into Edge dictionaries temporarily, filtering out non-existing edges
            valid_edges = []
            for edge_id in item_dict['node']['edges']:
                if edge_id in edge_dict:
                    valid_edges.append(edge_dict[edge_id])
                    valid_edge_count += 1  # increment the valid edge count
                else:
                    non_existing_edge_ids.append(edge_id)  # store non-existing edge ID
            item_dict['node']['edges'] = valid_edges  # replace with valid edges only
            node = Node(**item_dict['node'])
            data_to_export[k] = asdict(node)
        elif 'edge' in item_dict:
            edge = Edge(**item_dict['edge'])
            # Store the edge in the temporary dictionary with its ID
            edge_dict[edge.id] = asdict(edge)

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_export, json_file)

    print("Non-existing edge IDs:", len(non_existing_edge_ids))
    print("Valid edge count:", valid_edge_count)

    db.close()

# Example usage
db_filepath = 'preprocessing_pipeline/output_files/test_dict'
json_file_path = 'preprocessing_pipeline/output_files/db_export.json'
read_from_db(db_filepath, json_file_path)





