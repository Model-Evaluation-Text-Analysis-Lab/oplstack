import os
import uuid
import rocksdict
import json
import numpy as np
from pdfminer.high_level import extract_text
from typing import Dict, List, Optional, Union
from nltk.tokenize import sent_tokenize
import sentence_transformers
import fitz  # PyMuPDF
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
        edges = []  # TODO: Add edges generation logic here
        node = Node(
            id=node_id,
            content=chunk['content'],
            type=chunk['type'], 
            attributes=attributes,
            edges=edges,
            parent_id=prev_node_id  # Connect to the previous node
        )

        node_dict = {'node': asdict(node)}
        db[node_id] = json.dumps(node_dict).encode('utf-8')

        if prev_node_id is not None:
            parent_node_dict = json.loads(db[prev_node_id].decode('utf-8'))
            parent_node = Node(**parent_node_dict['node'])
            parent_node.edges.append(Edge(source=parent_node.id, target=node_id, type='child', attributes={}))
            db[prev_node_id] = json.dumps({'node': asdict(parent_node)}).encode('utf-8')

        prev_node_id = node_id
    db.close()

def embed_chunks(db_path, embeds_folder_path):
    db = rocksdict.Rdict(db_path)
    for k, v in db.items():
        node_dict = json.loads(v.decode('utf-8'))['node'] 
        node = Node(**node_dict)
        embedding = model.encode(node.content)
        np.save(f"{embeds_folder_path}/{k}.npy", embedding) 
        node_dict = {'node': asdict(node)}
        assert 'embedding' not in node_dict['node'].keys(), 'Embedding found in node dictionary.'
        
def embed_chunks_in_groups(db_path, embeds_folder_path):
    db = rocksdict.Rdict(db_path)
    for k, v in db.items():
        node_dict = json.loads(v.decode('utf-8'))['node']
        node = Node(**node_dict)
        
        # Append parent node content
        parent_id = node_dict.get('parent_id')
        if parent_id:
            parent_node = db.get(parent_id)
            if parent_node:
                parent_node_dict = json.loads(parent_node.decode('utf-8'))['node']
                parent_content = parent_node_dict.get('content', '')
                node.content = f"{parent_content}\n{node.content}"
        
        # Append child node content
        child_ids = [edge['target'] for edge in node_dict.get('edges', []) if edge['type'] == 'child']
        for child_id in child_ids:
            child_node = db.get(child_id)
            if child_node:
                child_node_dict = json.loads(child_node.decode('utf-8'))['node']
                child_content = child_node_dict.get('content', '')
                node.content = f"{node.content}\n{child_content}"
        
        embedding = model.encode(node.content)
        np.save(f"{embeds_folder_path}/{k}.npy", embedding) 
        node_dict = {'node': asdict(node)}
        assert 'embedding' not in node_dict['node'].keys(), 'Embedding found in node dictionary.'

