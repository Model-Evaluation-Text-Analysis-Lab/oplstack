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


def generate_and_display_images(pdf_path, all_words):
    with pdfplumber.open(pdf_path) as pdf:
        images = convert_from_path(pdf_path)

        for i, im in enumerate(images):
            print(f"Generating image for page {i+1}/{len(images)}...")

            words = [w for w in all_words if w['page'] == i+1]

            page_width = pdf.pages[i].width
            page_height = pdf.pages[i].height

            scale_x = im.width / page_width
            scale_y = im.height / page_height

            draw = ImageDraw.Draw(im)

            for word in words:
                x0, y0, x1, y1 = word['coordinates']

                x0 *= scale_x
                y0 *= scale_y
                x1 *= scale_x
                y1 *= scale_y

                draw.rectangle([x0, y0, x1, y1], outline="red")

            display(im)
            output_image_path = f"preprocessing_pipeline/output_files/PDF/pdfplumber_output_page_{i+1}.jpg"
            im.save(output_image_path, format="JPEG")
