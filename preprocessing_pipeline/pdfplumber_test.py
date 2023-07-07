#%%
import pdfplumber
from pdf2image import convert_from_path
from PIL import ImageDraw
from IPython.display import display
import json
import os
import uuid


#%%
def extract_words_from_pdf(pdf_path):
    # Open the PDF file
    all_words = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"Processing page {i+1}/{len(pdf.pages)}...")
            
            # Extract words
            words = page.extract_words(x_tolerance=2)

            # Refine the words into the desired format
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

#%%
def generate_and_display_images(pdf_path, all_words):
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Convert all pages to images
        images = convert_from_path(pdf_path)

        for i, im in enumerate(images):
            print(f"Generating image for page {i+1}/{len(images)}...")

            # Filter words for the current page
            words = [w for w in all_words if w['page'] == i+1]

            # Get the width and height of the current page
            page_width = pdf.pages[i].width
            page_height = pdf.pages[i].height

            # Calculate scaling factors
            scale_x = im.width / page_width
            scale_y = im.height / page_height

            # Create an ImageDraw object
            draw = ImageDraw.Draw(im)

            # Loop over the words and draw rectangles
            for word in words:
                # The word dictionary contains the coordinates of the bounding box
                x0, y0, x1, y1 = word['coordinates']

                # Scale the coordinates
                x0 *= scale_x
                y0 *= scale_y
                x1 *= scale_x
                y1 *= scale_y

                draw.rectangle([x0, y0, x1, y1], outline="red")

            # Display the image
            display(im)
            # Save the image to a file
            output_image_path = f"preprocessing_pipeline/output_files/PDF/pdfplumber_output_page_{i+1}.jpg"
            im.save(output_image_path, format="JPEG")

#%%
pdf_path = "preprocessing_pipeline/documents/complex.pdf"

#%%
all_words = extract_words_from_pdf(pdf_path)

#%%
generate_and_display_images(pdf_path, all_words)

#%%
# Convert the words object to a JSON string
words_json = json.dumps(all_words)

# Write the JSON string to a file
output_directory = "preprocessing_pipeline/output_files/PDF"
os.makedirs(output_directory, exist_ok=True)
with open('preprocessing_pipeline/output_files/PDF/pdfplumber_output.json', 'w') as f:
    f.write(words_json)

# %%
