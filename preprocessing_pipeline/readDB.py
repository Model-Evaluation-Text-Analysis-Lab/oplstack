import rocksdict
import json
from dataclasses import asdict
from datatypes import Node

def read_from_db(db_path, json_file_path):
    db = rocksdict.Rdict(db_path)
    data_to_export = {}

    for k, v in db.items():
        if 'node' in v:
            node = Node(**v['node'])
            data_to_export[k] = asdict(node)

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_export, json_file)

    db.close()

# Example usage
db_filepath = 'preprocessing_pipeline/output_files/test_dict'
json_file_path = 'preprocessing_pipeline/output_files/db_export.json'
read_from_db(db_filepath, json_file_path)
