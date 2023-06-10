import os
import uuid
import rocksdict
import json
import numpy as np
from pdfminer.high_level import extract_text
from typing import Dict, List, Optional, Union
from nltk.tokenize import sent_tokenize
import sentence_transformers
from datatypes import *
from dataclasses import asdict
import hashlib

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def generate_document_id(doc_name, doc_type):
    return hashlib.sha1((doc_name + doc_type).encode()).hexdigest()

def split_text_into_chunks(text, max_chunk_size=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_node_and_edge(db, node_id, content, node_type, attributes, parent_id):
    node = Node(
        id=node_id,
        content=content,
        type=node_type,
        attributes=attributes,
        edges=[],
        parent_id=parent_id  
    )
    db[node_id] = json.dumps({'node': asdict(node)}).encode('utf-8')

    # Create an edge from parent to the new node
    edge_id = str(uuid.uuid4())
    edge = Edge(
        id=edge_id,
        source=parent_id,
        target=node_id,
        type='child',
        attributes={}
    )
    db[edge_id] = json.dumps({'edge': asdict(edge)}).encode('utf-8')

    return node_id, edge_id

def update_parent_node(db, parent_id, child_edge_id):
    parent_dict = json.loads(db[parent_id].decode('utf-8'))
    parent = Node(**parent_dict['node'])
    parent.edges.append(child_edge_id)
    db[parent_id] = json.dumps({'node': asdict(parent)}).encode('utf-8')

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

def embed_chunks(db_path, embeds_folder_path, max_file_size_kb=500):
    os.makedirs(embeds_folder_path, exist_ok=True)
    db = rocksdict.Rdict(db_path)
    embeddings = []  # List to store embeddings
    file_number = 0  # Start with file number 0
    size_of_embedding = None
    for k, v in db.items():
        data_dict = json.loads(v.decode('utf-8'))
        if 'node' in data_dict:
            node = Node(**data_dict['node'])
            embedding = model.encode(node.content)
            if size_of_embedding is None:
                size_of_embedding = embedding.nbytes
            embeddings.append((k, embedding.tolist()))  # Store tuple of node id and embedding
            if len(embeddings) * size_of_embedding >= max_file_size_kb * 1024:  # Check the total size in bytes
                np.save(f"{embeds_folder_path}/embeddings{file_number}.npy", np.array(embeddings, dtype=object))  # Save embeddings
                embeddings = []  # Reset the embeddings list
                file_number += 1  # Increment the file number
    if embeddings:  # Save any remaining embeddings
        np.save(f"{embeds_folder_path}/embeddings{file_number}.npy", np.array(embeddings, dtype=object))
    db.close()
