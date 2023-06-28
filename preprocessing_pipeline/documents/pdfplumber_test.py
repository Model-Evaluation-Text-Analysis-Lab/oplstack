#%%
import pdfplumber
from pdf2image import convert_from_path
from PIL import ImageDraw
from IPython.display import display

#%%
def extract_words_from_pdf(pdf_path):
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Get the first page
        first_page = pdf.pages[1]

        # Extract words
        words = first_page.extract_words(x_tolerance=2)
    
    return words, first_page.width, first_page.height

#%%
def generate_and_display_image(pdf_path, words, page_width, page_height):
    # Convert the first page to an image
    images = convert_from_path(pdf_path, first_page=1, last_page=2)
    im = images[1]

    # Calculate scaling factors
    scale_x = im.width / page_width
    scale_y = im.height / page_height

    # Create a ImageDraw object
    draw = ImageDraw.Draw(im)

    # Loop over the words and draw rectangles
    for word in words:
        # The word dictionary contains the coordinates of the bounding box
        x0, y0, x1, y1 = word['x0'], word['top'], word['x1'], word['bottom']
        # Scale the coordinates
        x0 *= scale_x
        y0 *= scale_y
        x1 *= scale_x
        y1 *= scale_y
        draw.rectangle([x0, y0, x1, y1], outline="red")

    # Display the image
    display(im)

#%%
pdf_path = "preprocessing_pipeline/documents/complex.pdf"

#%%
words, page_width, page_height = extract_words_from_pdf(pdf_path)

#%%
generate_and_display_image(pdf_path, words, page_width, page_height)
