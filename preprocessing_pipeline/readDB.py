import rocksdict
import numpy as np
import json
from dataclasses import asdict
from datatypes import Root, Document, Node, Edge, Attribute

def read_from_db(db_path, json_file_path):
    db = rocksdict.Rdict(db_path)
    root_dict = db.get('root', asdict(Root(documents=[])))  # Get the root dictionary
    root = Root(**root_dict)  # Convert dictionary to Root instance

    data_to_export = {"root": {"documents": []}}

    for document_dict in root.documents:
        document = Document(**document_dict)  # Convert dictionary to Document instance
        doc_dict = {"id": document.id, "nodes": [], "edges": []}
        for node in document.nodes:
            node_instance = Node(**node)  # Convert dictionary to Node instance
            doc_dict["nodes"].append(asdict(node_instance))
        for edge in document.edges:
            edge_instance = Edge(**edge)  # Convert dictionary to Edge instance
            doc_dict["edges"].append(asdict(edge_instance))
        data_to_export["root"]["documents"].append(doc_dict)

    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_export, json_file)

    db.close()

# Example usage
db_filepath = 'preprocessing_pipeline/output_files/test_dict'
json_file_path = 'preprocessing_pipeline/output_files/db_export.json'
read_from_db(db_filepath, json_file_path)
