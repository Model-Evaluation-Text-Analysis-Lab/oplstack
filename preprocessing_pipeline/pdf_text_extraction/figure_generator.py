import os
'''import fitz
from tqdm import tqdm

def extract_figures_from_pdf(pdf_path, output_dir):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # For each page in the PDF
    for i in tqdm(range(len(doc)), desc="Extracting images from pages"):
        # For each image on the page
        for img in doc.get_page_images(i):
            # The image's identifier within the PDF
            xref = img[0]

            # Save the image as a PNG file
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:  # this means it's a CMYK image. Convert it to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            output_path = os.path.join(output_dir, f"figure_{i}_{xref}.png")
            pix.write_image(output_path)

            print(f'Saved figure {xref} of page {i} to {output_path}')


# Call the function with the paths to your PDF file and output directory
path_to_your_pdf_file = 'preprocessing_pipeline/documents/gzip.pdf'
path_to_your_output_dir = 'preprocessing_pipeline/output_files/PDF/figures'
extract_figures_from_pdf(path_to_your_pdf_file, path_to_your_output_dir)
'''

import fitz  # this is pymupdf
import io
import os
from PIL import Image

def extract_images_from_pdf(pdf_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF file with PyMuPDF
    doc = fitz.open(pdf_path)

    # Iterate over PDF pages
    for i in range(len(doc)):
        # Get the page
        page = doc[i]

        # Iterate over all images of the page
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            # Read the image data
            base_image = img[0]
            image_data = img[1]

            # Save the image data to a stream
            image_stream = io.BytesIO(image_data)

            # Read the image data from the stream
            image = Image.open(image_stream).convert("RGBA")

            # Construct the output image path
            image_path = os.path.join(output_dir, f'image_p{str(i)}_{str(image_index)}.png')

            # Save the image
            image.save(open(image_path, 'wb'))

            print(f'Saved image {image_index} on page {i} to {image_path}')

# Call the function with the paths to your PDF file and output directory
path_to_your_pdf_file = 'preprocessing_pipeline/documents/gzip.pdf'
path_to_your_output_dir = 'preprocessing_pipeline/output_files/PDF/figures'
extract_images_from_pdf(path_to_your_pdf_file, path_to_your_output_dir)


