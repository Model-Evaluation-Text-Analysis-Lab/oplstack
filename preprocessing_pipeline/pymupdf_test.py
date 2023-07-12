import fitz
from PIL import Image, ImageDraw
import json
import os
import uuid
from IPython.display import display


# Define the output directory
output_directory = "preprocessing_pipeline/output_files/PDF"
os.makedirs(output_directory, exist_ok=True)


def extract_words_from_pdf(pdf_path):
    all_words = []
    doc = fitz.open(pdf_path)
    
    for i, page in enumerate(doc):
        print(f"Processing page {i+1}/{len(doc)}...")

        blocks = page.get_text("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # this block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        text = s['text']
                        bbox = s['bbox']
                        refined_word = {
                            'uid': str(uuid.uuid4()),
                            'type': 'word',
                            'content': text,
                            'source': 'pymupdf',
                            'coordinates': [bbox[0], bbox[1], bbox[2], bbox[3]],
                            'page': i+1,
                        }
                        all_words.append(refined_word)

    return all_words


def generate_and_display_images(pdf_path, all_words):
    # Write the JSON string to a file
    output_directory = "preprocessing_pipeline/output_files/PDF"
    os.makedirs(output_directory, exist_ok=True)

    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        print(f"Generating image for page {i+1}/{len(doc)}...")

        # Convert page to pixmap
        pix = doc.get_page_pixmap(i)

        # Convert pixmap to PIL Image
        im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Filter words for the current page
        words = [w for w in all_words if w['page'] == i+1]

        # Create an ImageDraw object
        draw = ImageDraw.Draw(im)

        # Loop over the words and draw rectangles
        for word in words:
            # The word dictionary contains the coordinates of the bounding box
            x0, y0, x1, y1 = word['coordinates']

            draw.rectangle([x0, y0, x1, y1], outline="red")

        # Display the image
        display(im)
        # Save the image to a file
        output_image_path = f"{output_directory}/pymupdf_output_page_{i+1}.jpg"
        im.save(output_image_path, format="JPEG")


pdf_path = "preprocessing_pipeline/documents/complex.pdf"

all_words = extract_words_from_pdf(pdf_path)
generate_and_display_images(pdf_path, all_words)

# Convert the words object to a JSON string
words_json = json.dumps(all_words)

# Write the JSON string to a file
with open(f'{output_directory}/pymupdf_output.json', 'w') as f:
    f.write(words_json)

