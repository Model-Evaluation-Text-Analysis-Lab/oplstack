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
            # if i+1 <= 1:
            word_data, layout, layout_data = process_page(lp_model, ocr_agent, image, i)
            lp_all_word_data.extend(word_data)
            lp_all_layout_data.extend(layout_data)
            
            if generate_images:
                layout_svg = visualize_layout(image, layout, word_data, layout_data, i)

        with open('preprocessing_pipeline/output_files/PDF/lp_output_words.json', 'w') as f:
            json.dump(lp_all_word_data, f, indent=4)
        
        with open('preprocessing_pipeline/output_files/PDF/lp_output_layout.json', 'w') as f:
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

    all_word_data_pdfplumber = pdfplumber_all_words
    merged_boxes_pdfplumber, new_layout_data_pdfplumber = merge_boxes(all_word_data_pdfplumber, lp_all_layout_data, 'pdfplumber')

    all_word_data_pymupdf = pymupdf_all_words
    merged_boxes_pymupdf, new_layout_data_pymupdf = merge_boxes(all_word_data_pymupdf, lp_all_layout_data, 'pymupdf')

    with open('preprocessing_pipeline/output_files/PDF/merged_boxes_lp_pdfplumber.json', 'w') as f:
        json.dump(merged_boxes_pdfplumber, f, indent=4)
        
    with open('preprocessing_pipeline/output_files/PDF/new_layout_data_lp_pdfplumber.json', 'w') as f:
        json.dump(new_layout_data_pdfplumber, f, indent=4)
        
    with open('preprocessing_pipeline/output_files/PDF/merged_boxes_lp_pymupdf.json', 'w') as f:
        json.dump(merged_boxes_pymupdf, f, indent=4)
        
    with open('preprocessing_pipeline/output_files/PDF/new_layout_data_lp_pymupdf.json', 'w') as f:
        json.dump(new_layout_data_pymupdf, f, indent=4)

    pdf_tree_pdfplumber = generate_tree(new_layout_data_pdfplumber)
    with open('preprocessing_pipeline/output_files/PDF/pdf_tree_data_pdfplumber.json', 'w') as f:
        json.dump(pdf_tree_pdfplumber, f, indent=4)

    pdf_tree_pymupdf = generate_tree(new_layout_data_pymupdf)
    with open('preprocessing_pipeline/output_files/PDF/pdf_tree_data_pymupdf.json', 'w') as f:
        json.dump(pdf_tree_pymupdf, f, indent=4)

    return merged_boxes_pdfplumber, new_layout_data_pdfplumber, merged_boxes_pymupdf, new_layout_data_pymupdf

if __name__ == "__main__":
    merged_boxes_pdfplumber, new_layout_data_pdfplumber, merged_boxes_pymupdf, new_layout_data_pymupdf = process_pdf("preprocessing_pipeline/documents/complex.pdf", use_layoutparser=True, use_pdfplumber=True, use_pymupdf=False, generate_images=True)

