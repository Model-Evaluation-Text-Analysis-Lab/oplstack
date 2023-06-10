from pipeline_operations import split_text_into_chunks, store_chunks_in_db, embed_chunks
from document_loader import load_document
import os

document_filepaths = ['preprocessing_pipeline/documents/complex.pdf', 'preprocessing_pipeline/documents/sample.pdf'] 
db_filepath = 'preprocessing_pipeline/output_files/test_dict'
embeds_folder_path = 'preprocessing_pipeline/output_files/vector_store'

def execute_pipeline(document_filepaths, db_filepath, embeds_folder_path, max_chunk_size):
    for document_filepath in document_filepaths:
        text = load_document(document_filepath)
        if text:
            chunks = split_text_into_chunks(text, max_chunk_size)
            chunks_data = [{'content': chunk, 'type': 'text_chunk', 'attributes': {}} for chunk in chunks]  # Modify as needed
            store_chunks_in_db(chunks_data, db_filepath, document_filepath)
            embed_chunks(db_filepath, embeds_folder_path)
        else:
            print(f"Document {document_filepath} is empty. Skipping...")

if __name__ == '__main__':
    execute_pipeline(document_filepaths, db_filepath, embeds_folder_path, max_chunk_size=100)
