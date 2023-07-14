import time
from layout_parser_functions import init_models, extract_pages_and_images, process_page, visualize_layout
from pdfplumber_functions import extract_words_from_pdf, generate_and_display_images_pdfplumber
#from pymupdf_functions import extract_words_from_pdf_pymupdf, generate_and_display_images_pymupdf
from merge_boxes import merge_boxes, generate_tree
import json, os

def process_pdf(pdf_file_path, use_layoutparser=True, use_pdfplumber=True, use_pymupdf=True, generate_images=False):
    lp_all_word_data, pdfplumber_all_words, pymupdf_all_words = [], [], []
    lp_all_layout_data = []

    # Create output directories if they do not exist
    os.makedirs('preprocessing_pipeline/output_files/PDF', exist_ok=True)
    os.makedirs('preprocessing_pipeline/output_files/visuals', exist_ok=True)

    if use_layoutparser:
        start = time.time()
        lp_model, ocr_agent = init_models()
        lp_pdf_layout, lp_images = extract_pages_and_images(pdf_file_path)

        for i, image in enumerate(lp_images):
            print(f"Processing page {i+1}/{len(lp_images)} with LayoutParser...")
            _, word_data, layout, layout_data = process_page(lp_model, ocr_agent, image, i)
            lp_all_word_data.extend(word_data)
            lp_all_layout_data.extend(layout_data)
            
            if generate_images:
                layout_svg = visualize_layout(image, layout, word_data, i)

        with open('preprocessing_pipeline/output_files/PDF/lp_output_words.json', 'w') as f:
            json.dump(lp_all_word_data, f, indent=4)
        
        with open('preprocessing_pipeline/output_files/PDF/lp_output_words.json', 'w') as f:
            json.dump(lp_all_layout_data, f, indent=4)
        end = time.time()
        print("LayoutParser processing took", end - start, "seconds")

    if use_pdfplumber:
        start = time.time()
        print("Processing document with pdfplumber...")
        pdfplumber_all_words = extract_words_from_pdf(pdf_file_path)
        if generate_images:
            generate_and_display_images_pdfplumber(pdf_file_path, pdfplumber_all_words)

        with open('preprocessing_pipeline/output_files/PDF/pdfplumber_output.json', 'w') as f:
            json.dump(pdfplumber_all_words, f, indent=4)
        end = time.time()
        print("pdfplumber processing took", end - start, "seconds")

    if use_pymupdf:
        start = time.time()
        print("Processing document with PyMuPDF...")
        pymupdf_all_words = extract_words_from_pdf_pymupdf(pdf_file_path)
        if generate_images:
            generate_and_display_images_pymupdf(pdf_file_path, pymupdf_all_words)

        with open('preprocessing_pipeline/output_files/PDF/pymupdf_output.json', 'w') as f:
            json.dump(pymupdf_all_words, f, indent=4)
        end = time.time()
        print("PyMuPDF processing took", end - start, "seconds")

    all_word_data = pdfplumber_all_words
    merged_boxes, new_layout_data = merge_boxes(all_word_data, lp_all_layout_data)

    with open('preprocessing_pipeline/output_files/merged_boxes.json', 'w') as f:
        json.dump(merged_boxes, f, indent=4)
        
    with open('preprocessing_pipeline/output_files/new_layout_data.json', 'w') as f:
        json.dump(new_layout_data, f, indent=4)

    return merged_boxes, new_layout_data

if __name__ == "__main__":
    merged_boxes, merged_layout_data = process_pdf("preprocessing_pipeline/documents/complex.pdf", use_layoutparser=True, use_pdfplumber=True, use_pymupdf=False, generate_images=False)
    pdf_tree = generate_tree(merged_layout_data)
    with open('preprocessing_pipeline/output_files/pdf_tree_data.json', 'w') as f:
        json.dump(pdf_tree, f, indent=4)