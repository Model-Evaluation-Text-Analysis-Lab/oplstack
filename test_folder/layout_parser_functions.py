import layoutparser as lp
import json
from PIL import Image
import numpy as np
import cv2
import uuid
import os


def init_models():
    layout_model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6]
    )
    ocr_agent = lp.TesseractAgent(languages="eng")

    return layout_model, ocr_agent


def extract_pages_and_images(file_path):
    pdf_layout, images = lp.load_pdf(file_path, load_images=True)
    return pdf_layout, images


def process_page(layout_model, ocr_agent, image, i, confidence_threshold=0):
    layout = layout_model.detect(image)
    layout.sort(key=lambda block: (block.coordinates[0], block.coordinates[1]))
    page_data = []
    word_data = []
    layout_data = []
    for block in layout:
        segment_image = block.crop_image(image)
        res = ocr_agent.detect(
            segment_image,
            return_response=True,
            agg_output_level=lp.TesseractFeatureType.WORD
        )
        block_text = ""  # Initialize an empty string to store the OCR text for this block
        for index, row in res['data'].iterrows():
            if row['conf'] < confidence_threshold:
                continue
            block_text += " " + row['text']  # Append each word's text to the block_text

            x_1 = row['left']
            y_1 = row['top']
            x_2 = x_1 + row['width']
            y_2 = y_1 + row['height']

            # Add the top left corner coordinates of segment_image to each word's bounding box
            word_coordinates = (
                row.left + block.coordinates[0],
                row.top + block.coordinates[1],
                row.left + row.width + block.coordinates[0],
                row.top + row.height + block.coordinates[1]
            )
            word_info = {
                "uid": str(uuid.uuid4()),
                "type": "Word",
                "content": row['text'],
                "source": "lp",
                "coordinates": word_coordinates,
                "page": i+1  # Add the page number
            }
            word_data.append(word_info)

        layout_data.append({
            "type": block.type,
            "id": str(uuid.uuid4()),
            "source": "lp",
            "content": block_text.strip(),
            "coordinates": block.coordinates,
            "page": i+1  # Add the page number
        })

        page_data.extend(word_data)

    return page_data, word_data, layout, layout_data


def visualize_layout(image, layout, word_data):
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

        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    # draw the word level bounding boxes
    for word in word_data:
        x1, y1, x2, y2 = word['coordinates']
        color = (0, 165, 255)
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    return image
