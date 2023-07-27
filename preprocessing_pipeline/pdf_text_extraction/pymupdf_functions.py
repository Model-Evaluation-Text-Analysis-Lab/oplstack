import fitz
from PIL import Image, ImageDraw
import json
import os
import uuid

def extract_words_from_pdf_pymupdf(pdf_path):
    all_words = []
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        print(f"Processing page {i+1}/{len(doc)}...")

        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  
            if b['type'] == 0:  # this block contains text
                for l in b["lines"]:  
                    for s in l["spans"]:
                        # split the text in the span into words
                        for word in s['text'].split():
                            bbox = s['bbox']
                            refined_word = {
                                'uid': str(uuid.uuid4()),
                                'type': 'word',
                                'content': word,
                                'source': 'pymupdf',
                                'coordinates': [bbox[0], bbox[1], bbox[2], bbox[3]],
                                'page': i+1,
                                'size': s['size'],  # add size here
                            }
                            all_words.append(refined_word)

    return all_words


def generate_and_display_images_pymupdf(pdf_path, all_words):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        print(f"Generating image for page {i+1}/{len(doc)}...")

        pix = doc.get_page_pixmap(i)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        currentAxis = plt.gca()

        words = [w for w in all_words if w['page'] == i+1]

        for word in words:
            x0, y0, x1, y1 = word['coordinates']
            color = 'red'  
            currentAxis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=None, edgecolor=color, linewidth=2))

        output_filename = f'preprocessing_pipeline/output_files/visuals/pymupdf_output_page_{i+1}.pdf'
        plt.savefig(output_filename, bbox_inches='tight', pad_inches = 0)
        plt.close()