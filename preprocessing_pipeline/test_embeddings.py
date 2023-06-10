import glob
import numpy as np
import rocksdict, sys
from datatypes import *
import random
import json
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers
model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def get_node_content(database_path: str, node_id: str):
    database = rocksdict.Rdict(database_path)
    node = Node(**database[node_id]['node'])
    content = node.content
    database.close()
    return content

def test_decode_embed(index_file_path: str, database_path: str, embed_id: Optional[str] = None):
    with open(index_file_path, 'r') as json_file:
        index_list = json.load(json_file)
    if embed_id is None:
        random_node = random.choice(index_list)
        node_id = random_node['node_id']
    else:
        node_id = embed_id
    database = rocksdict.Rdict(database_path)
    node = Node(**database[node_id]['node'])
    print(node.content)
    database.close()

def similarity_search(search_embed, embeds_file_path):
    embeddings = np.load(embeds_file_path)
    with open(index_file_path, 'r') as json_file:
        index_list = json.load(json_file)
    similarities = cosine_similarity([search_embed], embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    top_node_ids = [index_list[i]['node_id'] for i in top_indices]
    return top_node_ids

db_filepath = 'preprocessing_pipeline/output_files/test_dict'
index_file_path = 'preprocessing_pipeline/output_files/vector_store/index_0.idx.json'
embeds_file_path = 'preprocessing_pipeline/output_files/vector_store/embeddings_0.npy'

search_term = "Everything about FJH"
search_embed = model.encode(search_term)

top_node_ids = similarity_search(search_embed, embeds_file_path)
search_results = [get_node_content(db_filepath, node_id) for node_id in top_node_ids]

search_dict = {search_term: search_results}
with open('preprocessing_pipeline/output_files/search_results.json', 'w') as json_file:
    json.dump(search_dict, json_file)
