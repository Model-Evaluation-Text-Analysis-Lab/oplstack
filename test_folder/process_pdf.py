import layout_parser_functions as lp_functions
import pdfplumber_functions as pdfplumber_functions
from merge_boxes import merge_boxes
import json
import numpy as np



def process_pdf(pdf_file_path):
    lp_model, ocr_agent = lp_functions.init_models()
    lp_pdf_layout, lp_images = lp_functions.extract_pages_and_images(pdf_file_path)

    lp_all_word_data = []
    lp_all_layout_data = []

    for i, image in enumerate(lp_images):
        print(f"Processing page {i+1}/{len(lp_images)}...")
        data, word_data, layout, layout_data = lp_functions.process_page(
            lp_model, ocr_agent, np.array(image), i
        )

        lp_all_word_data.extend(word_data)
        lp_all_layout_data.extend(layout_data)

    lp_layout_output_path = "preprocessing_pipeline/output_files/PDF/lp_output_layout.json"
    lp_words_output_path = "preprocessing_pipeline/output_files/PDF/lp_output_words.json"
    with open(lp_layout_output_path, 'w') as f:
        json.dump(lp_all_layout_data, f, indent=4)

    with open(lp_words_output_path, 'w') as f:
        json.dump(lp_all_word_data, f, indent=4)

    pdfplumber_all_words = pdfplumber_functions.extract_words_from_pdf(pdf_file_path)

    pdfplumber_output_path = "preprocessing_pipeline/output_files/PDF/pdfplumber_output.json"
    with open(pdfplumber_output_path, 'w') as f:
        json.dump(pdfplumber_all_words, f, indent=4)

    merged_boxes = merge_boxes(lp_all_word_data, pdfplumber_all_words)

    return merged_boxes


if __name__ == "__main__":
    process_pdf("preprocessing_pipeline/documents/complex.pdf")