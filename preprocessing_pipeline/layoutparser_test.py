#%%
import layoutparser as lp
import json
from PIL import Image
import numpy as np
from IPython.display import display
import cv2
import uuid
import os
ocr_agent = lp.TesseractAgent()

def initialize_models():
    layout_model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7]
    )
    ocr_agent = lp.ocr.TesseractAgent(languages="eng")

    return layout_model, ocr_agent

def process_block(block, image, ocr_agent):
    segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)
    text = ocr_agent.detect(segment_image)
    uid = str(uuid.uuid4())

    return {"uid": uid, "type": block.type, "content": text}

def extract_text_from_layout(layout, image, ocr_agent):
    data = []
    layout.sort(key=lambda block: (block.coordinates[0], block.coordinates[1]))

    for block in layout:
        data.append(process_block(block, image, ocr_agent))

    return data

def extract_page_layouts_from_pdf(file_path):
    pdf_layout, images = lp.load_pdf(file_path, load_images=True) 
    layout_model, ocr_agent = initialize_models()
    data = []
    vis_images = []
    for _, image in zip(pdf_layout, images):
        image = np.array(image)
        layout = layout_model.detect(image)
        page_data = extract_text_from_layout(layout, image, ocr_agent)
        data.extend(page_data)
        vis_images.append(visualize_layout(image, layout))
    return data, vis_images

def save_data_as_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def visualize_layout(image, layout):
    for block in layout:
        x1, y1, x2, y2 = block.coordinates        
        if block.type == "Text":
            color = (0, 255, 0) 
        elif block.type == "Title":
            color = (255, 0, 0) 
        elif block.type == "List":
            color = (0, 255, 255) 
        elif block.type == "Table":
            color = (255, 255, 0)
        elif block.type == "Figure":
            color = (0, 0, 255) 
        else:
            color = (0, 0, 0)
        
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    return image

# Specify the path to your file
input_file_path = "preprocessing_pipeline/documents/complex.pdf"
output_file_path = "preprocessing_pipeline/documents/sample_test_7.json"

data, vis_images = extract_page_layouts_from_pdf(input_file_path)

for vis_image in vis_images:
    display(Image.fromarray(vis_image))

save_data_as_json(data, output_file_path)

# %%
