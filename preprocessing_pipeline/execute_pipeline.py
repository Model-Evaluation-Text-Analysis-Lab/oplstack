from document_loader import split_text_into_chunks, load_document
from pipeline_operations import store_chunks_in_graph, embed_chunks
from readDB import readDB, write_to_file
from graph.graph import GraphDB, Vertex

document_filepaths = ['preprocessing_pipeline/documents/complex.pdf', 'preprocessing_pipeline/documents/sample.pdf']
db_filepath = 'preprocessing_pipeline/output_files/test_dict'
embeds_folder_path = 'preprocessing_pipeline/output_files/vector_store'

def execute_pipeline(document_filepaths, db_filepath, embeds_folder_path, max_chunk_size):
    graph_db = GraphDB(db_filepath)  # Instantiate GraphDB object here

    root_vertex = graph_db.get_vertex(Vertex.prefix + 'root')  # Check for root node in the db
    if not root_vertex:
        root_vertex = graph_db.add_vertex(type='root', content='root')  # Add root node if not exist

    for document_filepath in document_filepaths:
        text = load_document(document_filepath)
        if text:
            chunks = split_text_into_chunks(text, max_chunk_size)
            chunks_data = [{'content': chunk, 'type': 'text_chunk', 'attributes': {}} for chunk in chunks]  # Modify as needed
            store_chunks_in_graph(chunks_data, graph_db, document_filepath, root_vertex)  # Pass the GraphDB object to the function
            embed_chunks(graph_db, embeds_folder_path)  # Pass the GraphDB object here

        else:
            print(f"Document {document_filepath} is empty. Skipping...")

    graph_db.close_db()  # Close the database when done

if __name__ == '__main__':
    execute_pipeline(document_filepaths, db_filepath, embeds_folder_path, max_chunk_size=100)
    json_file_path = "preprocessing_pipeline/output_files/db_export.json"
    
    # Instantiate GraphDB object again before passing it to readDB
    graph_db = GraphDB(db_filepath)
    output = readDB(graph_db)
    write_to_file(output, json_file_path)
    graph_db.close_db()  # Be sure to close the database when done
