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
    source: str
    
def load_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        return load_pdf(file_path)
    elif file_extension == '.html':
        pass
        #return load_html(file_path)
    else:
        raise NotImplementedError(f"Loading documents of type {file_extension} is not supported.")

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
        
        node_dict = {'node': asdict(node)}
        db[node_id] = json.dumps(node_dict).encode('utf-8')

        if prev_node_id is not None:
            parent_node_dict = json.loads(db[prev_node_id].decode('utf-8'))
            parent_node = Node(**parent_node_dict['node'])
            parent_node.edges.append(node_id)
            db[prev_node_id] = json.dumps({'node': asdict(parent_node)}).encode('utf-8')

        prev_node_id = node_id

    db.close()

def embed_chunks(db_path, embeds_folder_path):
    
    db = rocksdict.Rdict(db_path)
    for k, v in db.items():
        node_dict = json.loads(v.decode('utf-8'))['node']  # no 'embed' in the dictionary anymore
        node = Node(**node_dict)

        embedding = model.encode(node.content)
        np.save(f"{embeds_folder_path}/{k}.npy", embedding)  # save embeddings as .npy files
        
        node_dict = {'node': asdict(node)}
        assert 'embedding' not in node_dict['node'].keys(), 'Embedding found in node dictionary.'

def execute_pipeline(document_filepath, db_filepath, embeds_folder_path):
    # 1. Load Document
    text = load_document(document_filepath)
    # 2. Index document into chunks
    chunks = split_text_into_chunks(text)
    # 3. Store grpah in rocsDB
    store_chunks_in_db(chunks, db_filepath, embeds_folder_path)
    # 4. Generate vector embeds fro chunks and store in vectore store
    embed_chunks(db_filepath, embeds_folder_path)

# Specify the file paths
document_filepath = 'preprocessing_pipeline/sample.pdf'
db_filepath = 'preprocessing_pipeline/test_dict'
embeds_folder_path = 'preprocessing_pipeline/vector_store'

# Execute the pipeline
execute_pipeline(document_filepath, db_filepath, embeds_folder_path)

