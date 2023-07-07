#%%
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
import json
import numpy as np

#%%
# Convert the PDF to an image using pdf2image (install with pip install pdf2image)
images = convert_from_path('preprocessing_pipeline/documents/complex.pdf')
img = images[0]  # Get the first page as an image

# Convert to RGB and get dimensions
img = img.convert('RGB')
width, height = img.size
# Create a blank image of the same dimensions as the PDF page
blank_image = np.ones((height, width, 3), np.uint8) * 255  # Set all pixels to white

# Open a matplotlib figure using the blank image
plt.figure(figsize=(10, 10))
plt.imshow(blank_image)
currentAxis = plt.gca()

# Calculate the scale factors
pdf_width_points, pdf_height_points = 612, 792
scale_x = width / pdf_width_points
scale_y = height / pdf_height_points

# Load your data
with open('preprocessing_pipeline/output_files/PDF/merged_output.json', 'r') as f:
    data = json.load(f)
    
# Load the bounding box data
with open('preprocessing_pipeline/output_files/PDF/lp_output_layout.json', 'r') as f:
    bounding_boxes = json.load(f)

# Apply the scale factors and draw the bounding boxes
for item in bounding_boxes:
    coordinates = [c * scale_x if i % 2 == 0 else c * scale_y for i, c in enumerate(item['coordinates'])]
    
    block_type = item['type']
    if block_type == "Text":
        color = 'green'
    elif block_type == "Title":
        color = 'red'
    elif block_type == "List":
        color = 'cyan'
    elif block_type == "Table":
        color = 'yellow'
    elif block_type == "Figure":
        color = 'blue'
    else:
        color = 'black'
    
    currentAxis.add_patch(
        plt.Rectangle(
            (coordinates[0], coordinates[1]),
            coordinates[2] - coordinates[0],
            coordinates[3] - coordinates[1],
            fill=None,
            edgecolor=color
        )
    )


# Get the dimensions of the original PDF in points (standard PDF unit, 1/72 inches)
pdf_width_points, pdf_height_points = 612, 792  # These are typical values for an A4 page, adjust them if your PDF size is different

# Get the scale factors
scale_x = width / pdf_width_points
scale_y = height / pdf_height_points

# Apply the scale factors when drawing boxes and adding text
for item in data:
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
# Save to a new SVG
plt.savefig('preprocessing_pipeline/output_files/PDF/merged_output.svg', format='svg', bbox_inches='tight', pad_inches = 0)
plt.savefig('preprocessing_pipeline/output_files/PDF/merged_output.pdf', bbox_inches='tight', pad_inches = 0)

# %%
