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

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

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

def store_chunks_in_db(data, db_path, embeds_folder_path):
    os.makedirs(embeds_folder_path, exist_ok=True)
    db = rocksdict.Rdict(db_path)
    prev_node_id = None
    for i, chunk in enumerate(data, start=1):
        node_id = str(uuid.uuid4())
        attributes = chunk['attributes']  # TODO: Add additional attributes extraction logic here
        edges = []  # Will contain edge IDs
        node = Node(
            id=node_id,
            content=chunk['content'],
            type=chunk['type'], 
            attributes=attributes,
            edges=edges,
            parent_id=prev_node_id  # Connect to the previous node
        )
        
        # Store the node
        node_dict = {'node': asdict(node)}
        db[node_id] = json.dumps(node_dict).encode('utf-8')

        # Create and store an edge if there is a previous node
        if prev_node_id is not None:
            edge_id = str(uuid.uuid4())
            edge = Edge(
                id=edge_id,
                source=prev_node_id,
                target=node_id,
                type='child',
                attributes={}
            )
            edge_dict = {'edge': asdict(edge)}
            db[edge_id] = json.dumps(edge_dict).encode('utf-8')

            # Update the previous node with the new edge
            parent_node_dict = json.loads(db[prev_node_id].decode('utf-8'))
            parent_node = Node(**parent_node_dict['node'])
            parent_node.edges.append(edge_id)
            db[prev_node_id] = json.dumps({'node': asdict(parent_node)}).encode('utf-8')

        prev_node_id = node_id

    db.close()
    
def embed_chunks(db_path, embeds_folder_path, max_file_size_kb=500):
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


