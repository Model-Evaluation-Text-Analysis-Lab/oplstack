import os
import uuid
import rocksdict
import json
import numpy as np
import pdftotext
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from nltk.tokenize import sent_tokenize
import sentence_transformers

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

@dataclass
class Node:
    id: str
    content: str
    attributes: Dict[str, 'Attribute']
    edges: List[str] = field(default_factory=list)

@dataclass
class Attribute:
    name: str
    value: str
    type: str

def load_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = pdftotext.PDF(file)
        text = "\n\n".join(pdf_reader)
        return text

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

def store_chunks_in_db(chunks, db_path, embeds_folder_path):
    os.makedirs(embeds_folder_path, exist_ok=True)  # create the folder if it doesn't exist
    db = rocksdict.Rdict(db_path)
    prev_node_id = None
    for i, chunk in enumerate(chunks, start=1):
        node_id = str(uuid.uuid4())
        attributes = {}  # replace this with your attributes extraction logic
        node = Node(
            id=node_id,
            content=chunk,
            attributes=attributes
        )
        embedding = model.encode(chunk)
        np.save(f"{embeds_folder_path}/{node_id}.npy", embedding)  # save embeddings as .npy files
        
        node_dict = {'node': asdict(node)}
        assert 'embedding' not in node_dict['node'].keys(), 'Embedding found in node dictionary.'
        
        db[node_id] = json.dumps(node_dict).encode('utf-8')

        if prev_node_id is not None:
            parent_node_dict = json.loads(db[prev_node_id].decode('utf-8'))
            parent_node = Node(**parent_node_dict['node'])
            parent_node.edges.append(node_id)
            db[prev_node_id] = json.dumps({'node': asdict(parent_node)}).encode('utf-8')

        prev_node_id = node_id

    db.close()

pdf_filepath = 'preprocessing_pipeline/sample.pdf'
db_filepath = 'preprocessing_pipeline/test_dict'
embeds_folder_path = 'preprocessing_pipeline/embeds_test'
text = load_pdf(pdf_filepath)
chunks = split_text_into_chunks(text)
store_chunks_in_db(chunks, db_filepath, embeds_folder_path)

