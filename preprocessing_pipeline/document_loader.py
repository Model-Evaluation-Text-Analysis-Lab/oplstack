from pipeline_operations import split_text_into_chunks
import os
import fitz  # PyMuPDF

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
        chunks = parse_pdf(file_path)
        return ' '.join(chunk['content'] for chunk in chunks) # Extract 'content' field from each dictionary
    elif file_extension == '.html':
        pass
        #return load_html(file_path)
    else:
        raise NotImplementedError(f"Loading documents of type {file_extension} is not supported.")

