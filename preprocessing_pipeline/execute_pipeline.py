from pipeline_operations import *
from document_loader import *

def execute_pipeline(document_filepath, db_filepath, embeds_folder_path):
    data = load_document(document_filepath)
    store_chunks_in_db(data, db_filepath, embeds_folder_path)
    embed_chunks_in_groups(db_filepath, embeds_folder_path)

document_filepath = 'preprocessing_pipeline/complex.pdf'
db_filepath = 'preprocessing_pipeline/test_dict'
embeds_folder_path = 'preprocessing_pipeline/vector_store'

execute_pipeline(document_filepath, db_filepath, embeds_folder_path)