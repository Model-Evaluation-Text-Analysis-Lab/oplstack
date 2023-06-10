import glob
import numpy as np

'''If the total number of embeddings is significantly larger than the 
number of unique embeddings, then there are duplicates in data'''

embeddings_files = glob.glob('preprocessing_pipeline/output_files/vector_store/*.npy')
embeddings = np.vstack([np.load(file) for file in embeddings_files])
print("Total embeddings:", len(embeddings))

unique_embeddings = np.unique(embeddings, axis=0)
print("Unique embeddings:", len(unique_embeddings))

def load_embeddings(embeds_folder_path, file_number):
    embeddings = np.load(f"{embeds_folder_path}/embeddings{file_number}.npy")
    for node_id, embedding in embeddings:
        print(f"Node ID: {node_id}")
        print(f"Embedding: {embedding}")

embeds_folder_path = 'preprocessing_pipeline/output_files/vector_store'
file_number = 0 # For the first file. Change it to inspect other files
load_embeddings(embeds_folder_path, file_number)
