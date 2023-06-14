import glob
import os
import uuid
import rocksdict, sys
import numpy as np
from typing import Dict, List, Optional, Union
import sentence_transformers
from datatypes import *
from dataclasses import asdict
import hashlib
import json

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def generate_document_id(doc_name, doc_type):
    return hashlib.sha1((doc_name + doc_type).encode()).hexdigest()

def create_node(db, node_id, content, node_type, attributes, parent_id=None):
    node = Node(
        id=node_id,
        content=content,
        type=node_type,
        attributes=attributes,
        edges=[],
        parent_id=parent_id,
    )
    db[node_id] = {'node': asdict(node)}
    return node_id

def create_edge(db, source_id, target_id):
    edge_id = source_id + "-parent-" + target_id + "-child"
    edge = Edge(
        id=edge_id,
        source=source_id,
        target=target_id,
        type='child',
        attributes={}
    )
    db[edge_id] = {'edge': asdict(edge)}
    return edge_id

def update_node_with_edge(db, node_id, edge_id):
    node = Node(**db[node_id]['node'])
    node.edges.append(edge_id)
    db[node_id] = {'node': asdict(node)}

def store_chunks_in_db(data, db_path, document_filepath):
    db = rocksdict.Rdict(db_path)

    # Check if root node exists, if not, create it
    if 'root' not in db:
        root_node_id = create_node(db, 'root', 'root', 'root', {})
    else:
        root_node_id = 'root'

    doc_name = os.path.basename(document_filepath)
    doc_type = os.path.splitext(doc_name)[1][1:]

    # Create or retrieve document type node under root node
    doc_type_node_id = generate_document_id(doc_type, 'type')
    if doc_type_node_id not in db:
        doc_type_node_id = create_node(db, doc_type_node_id, doc_type, "type", {}, parent_id=root_node_id)
        edge_id = create_edge(db, root_node_id, doc_type_node_id)
        update_node_with_edge(db, root_node_id, edge_id)

    # Create document node under document type node
    doc_node_id = generate_document_id(doc_name, doc_type)
    doc_node_id = create_node(db, doc_node_id, doc_name, "document", {}, parent_id=doc_type_node_id)
    edge_id = create_edge(db, doc_type_node_id, doc_node_id)
    update_node_with_edge(db, doc_type_node_id, edge_id)

    prev_node_id = doc_node_id

    # Add chunks under document node
    for i, chunk in enumerate(data, start=1):
        node_id = str(uuid.uuid4())
        attributes = chunk['attributes']
        attributes.update({'chunk_size': len(chunk['content'])})

        node_id = create_node(db, node_id, chunk['content'], chunk['type'], attributes, parent_id=prev_node_id)
        edge_id = create_edge(db, prev_node_id, node_id)
        
        update_node_with_edge(db, prev_node_id, edge_id)
        prev_node_id = node_id

    db.close()

def embed_chunks(database_path: str, embedding_folder_path: str, max_file_size_kb: int = 500) -> None:
    os.makedirs(embedding_folder_path, exist_ok=True)
    database = rocksdict.Rdict(database_path)
    embedding_list = []  # Initialize an empty list to store the embeddings
    index_list = []  # Initialize an empty list to store the indices

    # Determine the start of the file counter
    existing_files = glob.glob(os.path.join(embedding_folder_path, 'embeddings_*.npy'))
    if existing_files:
        file_counters = [int(os.path.splitext(os.path.basename(file))[0].split('_')[-1]) for file in existing_files]
        file_counter = max(file_counters) + 1
    else:
        file_counter = 0

    embedding_size = None

    for key, value in database.items():
        if 'node' in value:
            node = Node(**value['node'])
            embedding = model.encode(node.content)
            
            if embedding_size is None:
                embedding_size = sys.getsizeof(embedding)  # Calculate size of a single embedding in bytes
                
            embedding_list.append(embedding)
            index_entry = {'node_id': node.id, 'index': len(embedding_list) - 1}  # Store the node ID and the index
            
            # Check if the size of the embeddings list has exceeded the maximum file size
            if (len(embedding_list) * embedding_size) / 1024 > max_file_size_kb:  # Convert bytes to kilobytes
                np.save(os.path.join(embedding_folder_path, f'embeddings_{file_counter}.npy'), np.array(embedding_list))
                with open(os.path.join(embedding_folder_path, f'index_{file_counter}.idx.json'), 'w') as json_file:
                    json.dump(index_list, json_file)  # Save the entire index list for this file
                embedding_list = []  # Reset the list
                index_list = []  # Reset the index list
                file_counter += 1  # Increment the file counter
            else:
                index_list.append(index_entry)  # Only append to the index list if a new file isn't created
                
    # Store any remaining embeddings that didn't reach the maximum file size
    if embedding_list:
        np.save(os.path.join(embedding_folder_path, f'embeddings_{file_counter}.npy'), np.array(embedding_list))
        with open(os.path.join(embedding_folder_path, f'index_{file_counter}.idx.json'), 'w') as json_file:
            json.dump(index_list, json_file)  # Save the index list for the remaining embeddings
    
    # Debugging info
    print(f"Number of embeddings saved: {file_counter}")
    
    database.close()
