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

def create_node(db, node_id, content, node_type, attributes):
    node = Node(
        id=node_id,
        content=content,
        type=node_type,
        attributes=attributes,
    )
    db[node_id] = {'node': asdict(node)}
    return node

def create_edge(db, source_id, target_id):
    # Create an edge from source to target
    edge_id = source_id + "-parent-" + target_id + "-child"
    edge = Edge(
        id=edge_id,
        source=source_id,
        target=target_id,
        type='child',
    )
    db[edge_id] = {'edge': asdict(edge)}
    return edge

def store_chunks_in_db(data, db_path, document_filepath):
    db = rocksdict.Rdict(db_path)

    doc_name = os.path.basename(document_filepath)
    doc_type = os.path.splitext(doc_name)[1][1:]

    doc_node_id = generate_document_id(doc_name, doc_type)

    # Create nodes and edges for each chunk of data
    nodes = []
    edges = []
    prev_node = create_node(db, doc_node_id, doc_name, "document", {})
    nodes.append(prev_node)
    for i, chunk in enumerate(data, start=1):
        node_id = str(uuid.uuid4())
        attributes = chunk['attributes']
        attributes.update({'chunk_size': len(chunk['content'])})
        node = create_node(db, node_id, chunk['content'], chunk['type'], attributes)
        edge = create_edge(db, prev_node.id, node.id)
        nodes.append(node)
        edges.append(edge)
        prev_node = node

    # Create Document and store it under the root
    doc = Document(id=doc_node_id, nodes=nodes, edges=edges)
    
    root_dict = db.get('root', asdict(Root(documents=[])))  # Get the root dictionary
    root = Root(**root_dict)  # Convert dictionary to Root instance
    
    root.documents.append(doc)
    db['root'] = asdict(root)

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
                with open(os.path.join(embedding_folder_path, f'index_{file_counter}.idx.json'), 'w') as json_file:
                    json.dump(index_list, json_file)
                embedding_list = []  # Reset the list
                index_list = []  # Reset the list
                file_counter += 1  # Increment the file counter
                
    # Store any remaining embeddings that didn't reach the maximum file size
    if embedding_list:
        np.save(os.path.join(embedding_folder_path, f'embeddings_{file_counter}.npy'), np.array(embedding_list))
        with open(os.path.join(embedding_folder_path, f'index_{file_counter}.idx.json'), 'w') as json_file:
            json.dump(index_list, json_file)
        
    database.close()
