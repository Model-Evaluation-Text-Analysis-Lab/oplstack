import glob
from typing import Optional
import numpy as np
import random
import json
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers
from graph.graph import GraphDB

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def get_vertex_content(database_path: str, vertex_id: str):
    graph_db = GraphDB(database_path)
    vertex = graph_db.get_vertex(vertex_id)
    content = vertex.content
    graph_db.close_db()
    return content

def test_decode_embed(index_file_path: str, database_path: str, embed_id: Optional[str] = None):
    with open(index_file_path, 'r') as json_file:
        index_list = json.load(json_file)
    if embed_id is None:
        random_node = random.choice(index_list)
        vertex_id = random_node['vertex_id']
    else:
        vertex_id = embed_id
    graph_db = GraphDB(database_path)
    vertex = graph_db.get_vertex(vertex_id)
    print(vertex.content)
    graph_db.close_db()

def get_embeds_file_path(index_file_path: str, embeds_file_paths: list):
    index_file_name = index_file_path.split('/')[-1]  # Extract the index file name
    index_number = index_file_name.split('_')[1].split('.')[0]  # Extract the index number
    embeds_file_name = f"embeddings_{index_number}.npy"  # Generate the corresponding embeddings file name
    for embeds_file_path in embeds_file_paths:
        if embeds_file_name in embeds_file_path:
            return embeds_file_path
    raise ValueError(f"No matching embeddings file found for index file: {index_file_path}")


def similarity_search(search_embed, embeds_file_paths, index_file_paths):
    top_vertex_ids = []
    max_similarity = -1  # Initial maximum similarity
    for index_file_path in index_file_paths:
        with open(index_file_path, 'r') as json_file:
            index_list = json.load(json_file)
        embeds_file_path = get_embeds_file_path(index_file_path, embeds_file_paths)
        embeddings = np.load(embeds_file_path)
        similarities = cosine_similarity([search_embed], embeddings)[0]

        # Add these lines before the assertion
        print(len(embeddings), len(index_list))
        assert len(embeddings) == len(index_list), f"Embeddings and index list lengths do not match for file {embeds_file_path}"

        # Update maximum similarity if a higher value is found
        max_similarity = max(max_similarity, max(similarities))

        top_indices = similarities.argsort()[-3:][::-1]
        top_vertex_ids.extend([index_list[i]['vertex_id'] for i in top_indices])  # Use 'vertex_id' as the key

    print(f"Maximum similarity: {max_similarity}")
    return top_vertex_ids

db_filepath = 'preprocessing_pipeline/output_files/test_dict'
embeds_file_paths = sorted(glob.glob('preprocessing_pipeline/output_files/vector_store/embeddings_*.npy'))
index_file_paths = sorted(glob.glob('preprocessing_pipeline/output_files/vector_store/index_*.idx.json'))

search_term = "Mining"
search_embed = model.encode(search_term)

print(f"Number of embedding files: {len(embeds_file_paths)}")

top_vertex_ids = similarity_search(search_embed, embeds_file_paths, index_file_paths)
print(f"Number of retrieved vertices: {len(top_vertex_ids)}")

try:
    with open('preprocessing_pipeline/output_files/search_results.json', 'r') as json_file:
        search_dict = json.load(json_file)
except FileNotFoundError:
    search_dict = {}

# Retrieve previous search results for the current search term
previous_results = search_dict.get(search_term, [])
# Append the new search results
search_results = previous_results + [get_vertex_content(db_filepath, vertex_id) for vertex_id in top_vertex_ids]
# Update the search dictionary
search_dict[search_term] = search_results

# Save the updated search results
with open('preprocessing_pipeline/output_files/search_results.json', 'w') as json_file:
    json.dump(search_dict, json_file)
