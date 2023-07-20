import layoutparser as lp
from pdfplumber.utils import cluster_objects
import json
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
    return pdf_layout, [np.array(im) for im in images]


def process_page(layout_model, ocr_agent, image, i, confidence_threshold=0):
    layout = layout_model.detect(image)
    
    text_blocks = lp.Layout([b for b in layout if b.type in ['Text', 'Title', 'List']])
    figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
    
    text_blocks = lp.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
    
    h, w = image.shape[:2]

    # Determine the number of columns
    x_ranges = [(block.coordinates[0], block.coordinates[2]) for block in text_blocks]
    x_clusters = cluster_objects(x_ranges, key_fn=lambda x: sum(x)/2, tolerance=w/4)
    
    x_cluster_blocks = [[] for _ in range(len(x_clusters))]

    # assign each block to the closest cluster
    for block in text_blocks:
        x_center = (block.coordinates[0] + block.coordinates[2]) / 2
        min_distance = float('inf')
        closest_cluster_idx = None
        for idx, x_cluster in enumerate(x_clusters):
            cluster_center = sum(x_cluster[0]) / 2
            distance = abs(x_center - cluster_center)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_idx = idx
        x_cluster_blocks[closest_cluster_idx].append(block)

    # sort blocks in each cluster and combine all clusters into a single list
    column_blocks = []
    for cluster_blocks in x_cluster_blocks:
        cluster_blocks.sort(key=lambda b: b.coordinates[1])
        column_blocks.extend(cluster_blocks)

    text_blocks = lp.Layout(column_blocks)
    
    # And finally add the index according to the order
    text_blocks = lp.Layout([b.set(id=idx) for idx, b in enumerate(text_blocks)])
    
    page_data = []
    layout_data = []
    for block in text_blocks:
        segment_image = (block
                        .pad(left=5, right=5, top=5, bottom=5)
                        .crop_image(image))

        res = ocr_agent.detect(segment_image, return_response=True, agg_output_level=lp.TesseractFeatureType.WORD)

        block_text = ""  # Initialize an empty string to store the OCR text for this block
        word_data = []  # Initialize an empty list to store the word data for this block
        for index, row in res['data'].iterrows():
            if row['conf'] < confidence_threshold:
                continue
            block_text += " " + row['text']  # Append each word's text to the block_text

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
            "reading_order": block.id,
            "source": "lp",
            "content": block_text.strip(),
            "coordinates": block.coordinates,
            "page": i+1  # Add the page number
        })

        page_data.extend(word_data)

    return page_data, layout, layout_data



def visualize_layout(image, layout, word_data, layout_data, i):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    currentAxis = plt.gca()

    for block_data in layout_data:
        x1, y1, x2, y2 = block_data.get("coordinates")
        if block_data.get("type") == "Text":
            color = 'green'  # set the color in RGB format
        elif block_data.get("type") == "Title":
            color = 'red'
        elif block_data.get("type") == "List":
            color = 'cyan'
        elif block_data.get("type") == "Table":
            color = 'yellow'
        elif block_data.get("type") == "Figure":
            color = 'blue'
        else:
            color = 'black'
        currentAxis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, edgecolor=color, linewidth=2))
        
        reading_order = block_data.get("reading_order")
        if reading_order is not None:
            # Add a small rectangle with the reading order number inside it
            # We slightly offset the x and y coordinates to place the text inside the rectangle
            currentAxis.text(x1 + 10, y1 + 20, str(reading_order), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    
    # draw the word level bounding boxes
    # for word in word_data:
    #     x1, y1, x2, y2 = word['coordinates']
    #     color = 'orange'  # set the color in RGB format
    #     currentAxis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, edgecolor=color, linewidth=2))

    output_filename = f'preprocessing_pipeline/output_files/visuals/layout_visualization_page_{i+1}.pdf'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches = 0)
    plt.close()

    return output_filename

