import glob
from typing import Optional
import numpy as np
import rocksdict
import sys
import random
import json
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers
from datatypes import *

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

def similarity_search(search_embed, embeds_file_paths, index_file_paths):
    top_node_ids = []
    max_similarity = -1  # Initial maximum similarity
    for embeds_file_path, index_file_path in zip(embeds_file_paths, index_file_paths):
        embeddings = np.load(embeds_file_path)
        with open(index_file_path, 'r') as json_file:
            index_list = json.load(json_file)
        similarities = cosine_similarity([search_embed], embeddings)[0]

        # Make sure there are as many index entries as embeddings
        assert len(embeddings) == len(index_list), f"Embeddings and index list lengths do not match for file {embeds_file_path}"

        # Update maximum similarity if a higher value is found
        max_similarity = max(max_similarity, max(similarities))

        top_indices = similarities.argsort()[-3:][::-1]
        top_node_ids.extend([index_list[i]['node_id'] for i in top_indices])

    print(f"Maximum similarity: {max_similarity}")
    return top_node_ids


db_filepath = 'preprocessing_pipeline/output_files/test_dict'
embeds_file_paths = sorted(glob.glob('preprocessing_pipeline/output_files/vector_store/embeddings_*.npy'))
index_file_paths = sorted(glob.glob('preprocessing_pipeline/output_files/vector_store/index_*.idx.json'))

search_term = "urban mining"
search_embed = model.encode(search_term)

print(f"Number of embedding files: {len(embeds_file_paths)}")

top_node_ids = similarity_search(search_embed, embeds_file_paths, index_file_paths)
print(f"Number of retrieved nodes: {len(top_node_ids)}")

try:
    with open('preprocessing_pipeline/output_files/search_results.json', 'r') as json_file:
        search_dict = json.load(json_file)
except FileNotFoundError:
    search_dict = {}

# Retrieve previous search results for the current search term
previous_results = search_dict.get(search_term, [])
# Append the new search results
search_results = previous_results + [get_node_content(db_filepath, node_id) for node_id in top_node_ids]
# Update the search dictionary
search_dict[search_term] = search_results

# Save the updated search results
with open('preprocessing_pipeline/output_files/search_results.json', 'w') as json_file:
    json.dump(search_dict, json_file)
