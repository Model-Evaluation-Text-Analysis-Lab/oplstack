from document_loader import split_text_into_chunks, load_document
from pipeline_operations import store_chunks_in_graph, embed_chunks
from readDB import *

document_filepaths = ['preprocessing_pipeline/documents/complex.pdf', 'preprocessing_pipeline/documents/sample.pdf'] 
db_filepath = 'preprocessing_pipeline/output_files/test_dict'
embeds_folder_path = 'preprocessing_pipeline/output_files/vector_store'

def execute_pipeline(document_filepaths, db_filepath, embeds_folder_path, max_chunk_size):
    for document_filepath in document_filepaths:
        text = load_document(document_filepath)
        if text:
            chunks = split_text_into_chunks(text, max_chunk_size)
            chunks_data = [{'content': chunk, 'type': 'text_chunk', 'attributes': {}} for chunk in chunks]  # Modify as needed
            store_chunks_in_graph(chunks_data, db_filepath, document_filepath)
            embed_chunks(db_filepath, embeds_folder_path)
        else:
            print(f"Document {document_filepath} is empty. Skipping...")

if __name__ == '__main__':
    execute_pipeline(document_filepaths, db_filepath, embeds_folder_path, max_chunk_size=100)
    jjson_file_path_raw = "preprocessing_pipeline/output_files/db_export_raw.json"
    json_file_path_structured = "preprocessing_pipeline/output_files/db_export_structured.json"
    db_filepath = 'preprocessing_pipeline/output_files/test_dict'
    read_all_Edges_Vertices(db_filepath, jjson_file_path_raw)
    generate_json(db_filepath, json_file_path_structured)
