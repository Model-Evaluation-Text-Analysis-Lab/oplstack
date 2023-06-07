import os
import uuid
import rocksdict
import json
import numpy as np
from pdfminer.high_level import extract_text
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from nltk.tokenize import sent_tokenize
import sentence_transformers
import fitz  # PyMuPDF

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

@dataclass
class Attribute:
    name: str
    value: str
    type: str

@dataclass
class Edge:
    source: str
    target: str
    type: str
    attributes: Dict[str, Attribute]

@dataclass
class Node:
    id: str
    content: str
    type: str
    attributes: Dict[str, Attribute]
    edges: List[Edge]
    parent_id: Union[str, None] = None

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

def extract_text_from_pdf_page(page):
    """Extract and chunk text from a PDF page."""
    text_blocks = page.get_text("blocks")
    data = []
    for block in text_blocks:
        text = block[4]
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            if chunk.istitle():
                # This is a header
                data.append({"type": "text", "content": chunk, "attributes": {"format": "heading"}})
            else:
                # This is a paragraph
                data.append({"type": "text", "content": chunk, "attributes": {"format": "paragraph"}})
    return data

def extract_images_from_pdf_page(page):
    """Extract images from a PDF page."""
    # TODO: Extract images and save them into the images folder
    pass

def extract_tables_from_pdf_page(page):
    """Extract tables from a PDF page."""
    # TODO: Extract tables
    pass

def parse_pdf(file_path):
    """Parsing the PDF."""
    doc = fitz.open(file_path)
    data = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        data.extend(extract_text_from_pdf_page(page))
        extract_images_from_pdf_page(page)
        extract_tables_from_pdf_page(page)
    return data

def load_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        return parse_pdf(file_path)
    elif file_extension == '.html':
        pass
        #return load_html(file_path)
    else:
        raise NotImplementedError(f"Loading documents of type {file_extension} is not supported.")

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

def execute_pipeline(document_filepath, db_filepath, embeds_folder_path):
    data = load_document(document_filepath)
    store_chunks_in_db(data, db_filepath, embeds_folder_path)
    embed_chunks(db_filepath, embeds_folder_path)

document_filepath = 'preprocessing_pipeline/complex.pdf'
db_filepath = 'preprocessing_pipeline/test_dict'
embeds_folder_path = 'preprocessing_pipeline/vector_store'

execute_pipeline(document_filepath, db_filepath, embeds_folder_path)
