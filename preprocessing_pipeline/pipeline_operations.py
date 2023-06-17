from graph.graph import Vertex, Edge, GraphDB
import os
import hashlib
import sentence_transformers
import numpy as np
import sys
import glob
import re
import json

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def generate_document_id(doc_name, doc_type):
    return hashlib.sha1((doc_name + doc_type).encode()).hexdigest()

def store_chunks_in_graph(data, graph_db, document_filepath, root_vertex, vertex_cache):
    doc_name = os.path.basename(document_filepath)
    doc_type = os.path.splitext(doc_name)[1][1:]

    documents_vertex = get_or_create_vertex(graph_db, 'Documents', 'Documents', vertex_cache)
    graph_db.add_edge(root_vertex, documents_vertex, type1='child', type2='parent')

    doc_type_vertex = get_or_create_vertex(graph_db, doc_type, doc_type, vertex_cache)
    graph_db.add_edge(documents_vertex, doc_type_vertex, type1='child', type2='parent')

    doc_vertex = get_or_create_vertex(graph_db, doc_name, doc_name, vertex_cache)
    graph_db.add_edge(doc_type_vertex, doc_vertex, type1='child', type2='parent')

    prev_vertex = doc_vertex

    for i, chunk in enumerate(data, start=1):
        chunk_vertex = get_or_create_vertex(graph_db, chunk['type'], chunk['content'], vertex_cache)
        graph_db.add_edge(prev_vertex, chunk_vertex, type1='child', type2='parent')
        prev_vertex = chunk_vertex

def get_or_create_vertex(graph_db, type, content, vertex_cache):
    if (type, content) in vertex_cache:
        vertex = graph_db.get_vertex(vertex_cache[(type, content)])
    else:
        vertex = graph_db.add_vertex(type=type, content=content)
        vertex_cache[(type, content)] = vertex.id
    return vertex


# ----- EMBEDDINGS GENERATION -----

def create_embeddings(embedding_list, file_counter, embedding_folder_path):
    np.save(os.path.join(embedding_folder_path, f'embeddings_{file_counter}.npy'), np.array(embedding_list))

def save_indices(index_list, file_counter, embedding_folder_path):
    if len(index_list) == 0:
        print("Warning: Attempted to save an empty index list.")
    else:
        index_file_path = os.path.join(embedding_folder_path, f'index_{file_counter}.idx.json')
        with open(index_file_path, 'w') as json_file:
            json.dump(index_list, json_file)

def check_and_save_embeddings(embedding_list, index_list, embedding_size, max_file_size_kb, file_counter, embedding_folder_path):
    if (len(embedding_list) * embedding_size) / 1024 > max_file_size_kb:
        create_embeddings(embedding_list, file_counter, embedding_folder_path)
        save_indices(index_list, file_counter, embedding_folder_path)
        return [], [], file_counter + 1
    return embedding_list, index_list, file_counter

def embed_chunks(graph_db: GraphDB, embedding_folder_path: str, max_file_size_kb: int = 500) -> None:
    os.makedirs(embedding_folder_path, exist_ok=True)
    # graph_db = GraphDB(database_path)
    embedding_list = []  # Initialize an empty list to store the embeddings
    index_list = []  # Initialize an empty list to store the indices

    existing_files = glob.glob(os.path.join(embedding_folder_path, 'index_*.idx.json'))
    if existing_files:
        # Extract numbers from filenames using regular expressions
        file_counters = [int(re.findall(r'\d+', os.path.basename(file))[0]) for file in existing_files]
        file_counter = max(file_counters) + 1

        # Load the latest processed vertices
        with open(max(existing_files, key=os.path.getctime), 'r') as f:
            processed_vertices = {entry['vertex_id'] for entry in json.load(f)}
    else:
        file_counter = 0
        processed_vertices = set()

    embedding_size = None

    for key, content in graph_db.db.items():
        if key.startswith(Vertex.prefix) and key not in processed_vertices:
            embedding = model.encode(content)
            
            if embedding_size is None:
                embedding_size = sys.getsizeof(embedding)  # Calculate size of a single embedding in bytes
                
            embedding_list.append(embedding)
            index_entry = {'vertex_id': key, 'index': len(embedding_list) - 1}
            index_list.append(index_entry)  # Add index_entry to index_list before checking and saving embeddings
            embedding_list, index_list, file_counter = check_and_save_embeddings(embedding_list, index_list, embedding_size, max_file_size_kb, file_counter, embedding_folder_path)

    if embedding_list:
        create_embeddings(embedding_list, file_counter, embedding_folder_path)
        save_indices(index_list, file_counter, embedding_folder_path)

    print(f"Number of embeddings saved: {file_counter}")

    # graph_db.close_db()
