import os
import uuid
import rocksdict, sys
import numpy as np
from pdfminer.high_level import extract_text
from typing import Dict, List, Optional, Union
from nltk.tokenize import sent_tokenize
import sentence_transformers
from datatypes import *
from dataclasses import asdict
import hashlib
import json


model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def generate_document_id(doc_name, doc_type):
    return hashlib.sha1((doc_name + doc_type).encode()).hexdigest()

def create_node_and_edge(db, node_id, content, node_type, attributes, parent_id):
    node = Node(
        id=node_id,
        content=content,
        type=node_type,
        attributes=attributes,
        edges=[],
        parent_id=parent_id  
    )
    db[node_id] = {'node': asdict(node)}

    # Create an edge from parent to the new node
    edge_id = str(uuid.uuid4())
    edge = Edge(
        id=edge_id,
        source=parent_id,
        target=node_id,
        type='child',
        attributes={}
    )
    db[edge_id] = {'edge': asdict(edge)}

    return node_id, edge_id

def update_parent_node(db, parent_id, child_edge_id):
    parent = Node(**db[parent_id]['node'])
    parent.edges.append(child_edge_id)
    db[parent_id] = {'node': asdict(parent)}

def store_chunks_in_db(data, db_path, document_filepath):
    db = rocksdict.Rdict(db_path)

    root_id = 'root'  
    if root_id not in db:
        create_node_and_edge(db, root_id, "Root", "root", {}, None)

    doc_name = os.path.basename(document_filepath)
    doc_type = os.path.splitext(doc_name)[1][1:]
    type_node_id = doc_type

    # Check if the file type node already exists
    if type_node_id not in db:
        type_node_id, root_type_edge_id = create_node_and_edge(db, type_node_id, doc_type, "file_type", {}, root_id)
        update_parent_node(db, root_id, root_type_edge_id)

    doc_node_id = generate_document_id(doc_name, doc_type)
    doc_node_id, type_doc_edge_id = create_node_and_edge(db, doc_node_id, doc_name, "document", {}, type_node_id)
    update_parent_node(db, type_node_id, type_doc_edge_id)

    prev_node_id = doc_node_id 

    for i, chunk in enumerate(data, start=1):
        node_id = str(uuid.uuid4())
        attributes = chunk['attributes']
        attributes.update({'chunk_size': len(chunk['content'])})

        node_id, child_edge_id = create_node_and_edge(db, node_id, chunk['content'], chunk['type'], attributes, prev_node_id)
        if prev_node_id is not None:
            update_parent_node(db, prev_node_id, child_edge_id)

        prev_node_id = node_id

    db.close()

def embed_chunks(database_path: str, embedding_folder_path: str, max_file_size_kb: int = 500) -> None:
    os.makedirs(embedding_folder_path, exist_ok=True)
    database = rocksdict.Rdict(database_path)
    embedding_list = []  # Initialize an empty list to store the embeddings
    index_list = []  # Initialize an empty list to store the node ID and index information
    file_counter = 0  # Initialize a counter to keep track of the file number
    embedding_size = None

    for key, value in database.items():
        if 'node' in value:
            node = Node(**value['node'])
            embedding = model.encode(node.content)
            
            if embedding_size is None:
                embedding_size = sys.getsizeof(embedding)  # Calculate size of a single embedding in bytes
                
            embedding_list.append(embedding)
            index_list.append({'node_id': node.id, 'index': len(embedding_list) - 1})  # Store the node ID and the index
            
            # Check if the size of the embeddings list has exceeded the maximum file size
            if (len(embedding_list) * embedding_size) / 1024 > max_file_size_kb:  # Convert bytes to kilobytes
                np.save(os.path.join(embedding_folder_path, f'embeddings_{file_counter}.npy'), np.array(embedding_list))
                with open(os.path.join(embedding_folder_path, f'pdf_document_name_{file_counter}.idx.json'), 'w') as json_file:
                    json.dump(index_list, json_file)
                embedding_list = []  # Reset the list
                index_list = []  # Reset the list
                file_counter += 1  # Increment the file counter
                
    # Store any remaining embeddings that didn't reachs the maximum file size
    if embedding_list:
        np.save(os.path.join(embedding_folder_path, f'embeddings_{file_counter}.npy'), np.array(embedding_list))
        with open(os.path.join(embedding_folder_path, f'embed_index_{file_counter}.idx.json'), 'w') as json_file:
            json.dump(index_list, json_file)
        
    database.close()