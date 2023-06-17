import layoutparser as lp
import json
from PIL import Image
import numpy as np
ocr_agent = lp.TesseractAgent()

def extract_text_from_pdf_page(layout, image):
    # Initialize the layout detection model
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
    )

    # Detect the layout of the page and initialize the OCR agent
    layout = model.detect(image)
    ocr_agent = lp.ocr.TesseractAgent(languages="eng")

    # Extract the text from the layout
    data = []
    for block in layout:
        # Extract the text from the block
        segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)
        text = ocr_agent.detect(segment_image)

        # Append the text to the data list
        data.append({"type": block.type, "content": text})

    return data


def extract_text_from_pdf(file_path):
    pdf_layout, images = lp.load_pdf(file_path, load_images=True) # Load images along with layout
    data = []
    for layout, image in zip(pdf_layout, images):
        # Convert the PIL Image to OpenCV array
        image = np.array(image)
        page_data = extract_text_from_pdf_page(layout, image)
        data.extend(page_data)
    return data

def save_data_as_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Specify the path to your file
input_file_path = "preprocessing_pipeline/documents/sample.pdf"
output_file_path = "preprocessing_pipeline/documents/sample.json"

data = extract_text_from_pdf(input_file_path)
save_data_as_json(data, output_file_path)



'''import cv2
import layoutparser as lp

# Load the image
image_path = "preprocessing_pipeline/documents/image.jpeg"
image = cv2.imread(image_path)

# Initialize the layout model
model = lp.Detectron2LayoutModel(
    config_path ='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    label_map ={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
)

# Use the model to detect layout elements
layout = model.detect(image)
'''