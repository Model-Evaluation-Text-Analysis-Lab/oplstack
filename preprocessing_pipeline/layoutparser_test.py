#%%
import layoutparser as lp
import json
from PIL import Image
import numpy as np
from IPython.display import display
import cv2
import uuid
ocr_agent = lp.TesseractAgent()

#%%
def initialize_models():
    layout_model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
    )
    ocr_agent = lp.ocr.TesseractAgent(languages="eng")

    return layout_model, ocr_agent

#%%
def process_block(block, image, ocr_agent, stack, orphaned_blocks):
    print(f"Processing block with type {block.type}")
    print(f"Block coordinates: {block.coordinates}")

    segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)
    text = ocr_agent.detect(segment_image)

    uid = str(uuid.uuid4())
    print(f"Generated uid for block: {uid}")

    if block.type == "Title":
        print(f"Identified a title block with uid {uid} and content: {text}")

        children = []
        while stack and stack[-1]["type"] != "Title":
            children.append(stack.pop())

        # Also add any orphaned blocks to the list of children
        children.extend(orphaned_blocks)
        orphaned_blocks = []

        if stack:
            title_block = stack.pop()
            title_block["children"].extend(children)
            stack.append(title_block)

        stack.append({"uid": uid, "type": block.type, "content": text, "children": []})

    else:
        orphaned_blocks.append({"uid": uid, "type": block.type, "content": text})

    return stack, orphaned_blocks

#%%
def extract_text_from_layout(layout, image, ocr_agent):
    data = []
    stack = []
    orphaned_blocks = []

    layout.sort(key=lambda block: (block.coordinates[0], block.coordinates[1]))
    print("Sorted layout blocks by their position in the document")

    for block in layout:
        stack, orphaned_blocks = process_block(block, image, ocr_agent, stack, orphaned_blocks)

    while stack:
        children = []
        while stack and stack[-1]["type"] != "Title":
            children.append(stack.pop())

        # Also add any remaining orphaned blocks to the list of children
        children.extend(orphaned_blocks)
        orphaned_blocks = []

        if stack:
            title_block = stack.pop()
            title_block["children"].extend(children)
            data.append(title_block)

    return data


#%%
def extract_text_from_pdf_page(image):
    layout_model, ocr_agent = initialize_models()

    layout = layout_model.detect(image)

    data = extract_text_from_layout(layout, image, ocr_agent)

    vis_image = visualize_layout(image, layout)

    return data, vis_image

#%%
def extract_text_from_pdf(file_path):
    pdf_layout, images = lp.load_pdf(file_path, load_images=True) # Load images along with layout
    data = []
    vis_images = []
    for _, image in zip(pdf_layout, images):
        # Convert the PIL Image to OpenCV array
        image = np.array(image)
        page_data, vis_image = extract_text_from_pdf_page(image)
        data.extend(page_data)
        vis_images.append(vis_image)
    return data, vis_images

#%%
def save_data_as_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

#%%
def visualize_layout(image, layout):
    for block in layout:
        # Get the coordinates of the block
        x1, y1, x2, y2 = block.coordinates        
        # Map the block type to a color
        if block.type == "Text":
            color = (0, 255, 0) # Green
        elif block.type == "Title":
            color = (255, 0, 0) # Red
        elif block.type == "List":
            color = (0, 255, 255) 
        elif block.type == "Table":
            color = (255, 255, 0)
        elif block.type == "Figure":
            color = (0, 0, 255) # Blue
        else:
            color = (0, 0, 0)
        
        # Draw the bounding box on the image
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    return image

#%%
# Specify the path to your file
input_file_path = "preprocessing_pipeline/documents/complex.pdf"
output_file_path = "preprocessing_pipeline/documents/sample_test_7.json"

#%%
data, vis_images = extract_text_from_pdf(input_file_path)

#%%
for vis_image in vis_images:
    display(Image.fromarray(vis_image))

#%%
print(data)

#%%
save_data_as_json(data, output_file_path)