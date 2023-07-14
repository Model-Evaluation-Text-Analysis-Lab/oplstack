#%%
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
from PyPDF2 import PdfMerger
import json
import numpy as np
import os

#%%
# Convert the PDF to images using pdf2image
images = convert_from_path('preprocessing_pipeline/documents/complex.pdf')

# Get PDF dimensions
pdf_width_points, pdf_height_points = 612, 792

# Load your merged data
with open('preprocessing_pipeline/output_files/merged_boxes.json', 'r') as f:
    data = json.load(f)

# Group data by page
data_by_page = {}
for item in data:
    page = item['page']
    if page not in data_by_page:
        data_by_page[page] = []
    data_by_page[page].append(item)

# Load the bounding box data
with open('preprocessing_pipeline/output_files/PDF/lp_output_words.json', 'r') as f:
    bounding_boxes = json.load(f)

# Group bounding box data by page
bounding_boxes_by_page = {}
for item in bounding_boxes:
    page = item['page']
    if page not in bounding_boxes_by_page:
        bounding_boxes_by_page[page] = []
    bounding_boxes_by_page[page].append(item)

# Load the bounding box data from PyMuPDF
# with open('preprocessing_pipeline/output_files/PDF/pymupdf_output.json', 'r') as f:
#     pymupdf_bounding_boxes = json.load(f)

# # Group bounding box data by page for PyMuPDF
# pymupdf_bounding_boxes_by_page = {}
# for item in pymupdf_bounding_boxes:
#     page = item['page']
#     if page not in pymupdf_bounding_boxes_by_page:
#         pymupdf_bounding_boxes_by_page[page] = []
#     pymupdf_bounding_boxes_by_page[page].append(item)


# Create a PdfMerger object
merger = PdfMerger()

# Loop over each page
for page_number, img in enumerate(images, start=1):
    img = img.convert('RGB')
    width, height = img.size

    # Create a blank image of the same dimensions as the PDF page
    blank_image = np.ones((height, width, 3), np.uint8) * 255  # Set all pixels to white

    # Open a matplotlib figure using the blank image
    plt.figure(figsize=(10, 10))
    plt.imshow(blank_image)
    currentAxis = plt.gca()

    # Calculate the scale factors
    scale_x = width / pdf_width_points
    scale_y = height / pdf_height_points

    # Get the data for this page
    page_data = data_by_page[page_number]

    # Get the bounding boxes for this page
    page_bounding_boxes = bounding_boxes_by_page[page_number]

    # Get the bounding boxes for this page from PyMuPDF
    # page_pymupdf_bounding_boxes = pymupdf_bounding_boxes_by_page[page_number]

    # # Draw the bounding boxes from PyMuPDF
    # for item in page_pymupdf_bounding_boxes:
    #     coordinates = [c * scale_x if i % 2 == 0 else c * scale_y for i, c in enumerate(item['coordinates'])]
    #     block_type = item['type']
    #     color = 'green'  # Change the colors as needed for PyMuPDF
    #     currentAxis.add_patch(
    #         plt.Rectangle((coordinates[0], coordinates[1]), coordinates[2] - coordinates[0], coordinates[3] - coordinates[1],
    #                     fill=None, edgecolor=color, linewidth=2))

    # Draw the bounding boxes
    for item in page_bounding_boxes:
        coordinates = [c * scale_x if i % 2 == 0 else c * scale_y for i, c in enumerate(item['coordinates'])]
        block_type = item['type']
        color = 'red' if block_type == "Text" else 'blue'  # Change the colors as needed
        currentAxis.add_patch(
            plt.Rectangle((coordinates[0], coordinates[1]), coordinates[2] - coordinates[0], coordinates[3] - coordinates[1],
                          fill=None, edgecolor=color, linewidth=2))

    # Apply the scale factors when drawing boxes and adding text
    for item in page_data:
        coords = [c * scale_x if i % 2 == 0 else c * scale_y for i, c in enumerate(item['coordinates'])]
        content = item['content']
        original_content =  item['discarded_content']
        box_width = coords[2] - coords[0]
        box_height = coords[3] - coords[1]
        currentAxis.add_patch(
            plt.Rectangle((coords[0], coords[1]), box_width, box_height, fill=None, edgecolor='black'))

        # Calculate the position of the text
        text_x = coords[0] + box_width * 0.05  # Adjust the offset to position the text horizontally inside the box
        text_y = coords[1] + box_height * 0.5  # Adjust the offset to position the text vertically inside the box

        # Add the original content just above the current content
        plt.text(text_x, text_y - box_height * 0.2, original_content, fontsize=2, color='purple')

        # Add the current content
        plt.text(text_x, text_y, content, fontsize=4, color='blue')

    plt.axis('off')
    # Save each page to a new PDF
    output_filename = f'preprocessing_pipeline/output_files/PDF/merged_output_page_{page_number}.pdf'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches = 0)
    plt.close()  # Close the figure to free up memory

    # Add the page to the PdfFileMerger
    merger.append(output_filename)

# Write the merged PDF to a file
merger.write("preprocessing_pipeline/output_files/PDF/merged_output.pdf")

# Delete the individual page PDFs
for page_number in range(1, len(images) + 1):
    os.remove(f'preprocessing_pipeline/output_files/PDF/merged_output_page_{page_number}.pdf')

# Close the PdfFileMerger
merger.close()
