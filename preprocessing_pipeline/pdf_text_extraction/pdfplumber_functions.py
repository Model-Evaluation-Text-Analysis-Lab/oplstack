import pdfplumber
from pdf2image import convert_from_path
from PIL import ImageDraw
import json
import os
import uuid

def extract_words_from_pdf(pdf_path):
    all_words = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"Processing page {i+1}/{len(pdf.pages)}...")

            words = page.extract_words(x_tolerance=2)

            refined_words = []
            for word in words:
                refined_word = {
                    'uid': str(uuid.uuid4()),
                    'type': 'word',
                    'content': word['text'],
                    'source': 'pdfplumber',
                    'coordinates': [word['x0'], word['top'], word['x1'], word['bottom']],
                    'page': i+1,
                }
                refined_words.append(refined_word)

            all_words.extend(refined_words)

    return all_words


def generate_and_display_images_pdfplumber(pdf_path, all_words):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    with pdfplumber.open(pdf_path) as pdf:
        images = convert_from_path(pdf_path)

        for i, im in enumerate(images):
            print(f"Generating image for page {i+1}/{len(images)}...")

            page_width = pdf.pages[i].width
            page_height = pdf.pages[i].height

            scale_x = im.width / page_width
            scale_y = im.height / page_height

            plt.figure(figsize=(10, 10))
            plt.imshow(im)
            currentAxis = plt.gca()

            words = [w for w in all_words if w['page'] == i+1]

            for word in words:
                x0, y0, x1, y1 = word['coordinates']
                x0 *= scale_x
                y0 *= scale_y
                x1 *= scale_x
                y1 *= scale_y

                color = 'red'  
                currentAxis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=None, edgecolor=color, linewidth=2))

            output_filename = f'preprocessing_pipeline/output_files/visuals/pdfplumber_output_page_{i+1}.pdf'
            plt.savefig(output_filename, bbox_inches='tight', pad_inches = 0)
            plt.close()

