import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import uuid

def merge_boxes(all_word_data, layout_data):
    boxes_by_page = {}
    for box in all_word_data:
        page = box['page']
        if page not in boxes_by_page:
            boxes_by_page[page] = []
        boxes_by_page[page].append(box)

    new_boxes = []
    new_layout_data = []

    for page, boxes in boxes_by_page.items():
        boxes_np = np.array([box['coordinates'] for box in boxes])
        boxes_idx = {idx: box['uid'] for idx, box in enumerate(boxes)}

        x_max = np.maximum(boxes_np[:, None, 0], boxes_np[:, 0])
        y_max = np.maximum(boxes_np[:, None, 1], boxes_np[:, 1])
        x_min = np.minimum(boxes_np[:, None, 2], boxes_np[:, 2])
        y_min = np.minimum(boxes_np[:, None, 3], boxes_np[:, 3])

        interArea = np.maximum(0, x_min - x_max + 1) * np.maximum(0, y_min - y_max + 1)
        boxAArea = (boxes_np[:, 2] - boxes_np[:, 0] + 1) * (boxes_np[:, 3] - boxes_np[:, 1] + 1)
        boxBArea = (boxes_np[:, None, 2] - boxes_np[:, None, 0] + 1) * (boxes_np[:, None, 3] - boxes_np[:, None, 1] + 1)
        iou = interArea / (boxAArea + boxBArea - interArea)

        graph = (iou >= 0.1).astype(int)
        graph_csr = csr_matrix(graph)

        n_components, labels = connected_components(
            csgraph=graph_csr,
            directed=False,
            return_labels=True
        )

        for label in range(n_components):
            indices = np.where(labels == label)[0]

            content = ''
            discarded_content = ''
            coordinates = []

            for idx in indices:
                box = boxes[idx]
                if box['source'] == 'pdfplumber':
                    content += ' ' + box['content']
                else:
                    discarded_content += ' ' + box['content']

                coordinates.append(box['coordinates'])

            x_min = min(coordinate[0] for coordinate in coordinates)
            y_min = min(coordinate[1] for coordinate in coordinates)
            x_max = max(coordinate[2] for coordinate in coordinates)
            y_max = max(coordinate[3] for coordinate in coordinates)
            union_coordinates = [x_min, y_min, x_max, y_max]

            new_boxes.append({
                'uid': str(uuid.uuid4()),
                'type': 'Word',
                'content': content.strip(),
                'discarded_content': discarded_content.strip(),
                'coordinates': union_coordinates,
                'page': page
            })

        # Construct new layout data
        for layout in layout_data:
            if layout['page'] != page:
                continue

            layout_content = ""
            layout_box = layout['coordinates']
            for box in new_boxes:
                box_coordinates = box['coordinates']
                if max(layout_box[0], box_coordinates[0]) <= min(layout_box[2], box_coordinates[2]) and \
                max(layout_box[1], box_coordinates[1]) <= min(layout_box[3], box_coordinates[3]):
                    layout_content += " " + box['content']

            new_layout_data.append({
                "type": layout['type'],
                "id": str(uuid.uuid4()),
                "source": "lp",
                "content": layout_content.strip(),
                "coordinates": layout_box,
                "page": page  # Add the page number
            })

        # Handle non-overlapping boxes
        non_overlapping_boxes = [box for box in new_boxes if not any(
            max(layout_box['coordinates'][0], box['coordinates'][0]) <= min(layout_box['coordinates'][2], box['coordinates'][2]) and \
            max(layout_box['coordinates'][1], box['coordinates'][1]) <= min(layout_box['coordinates'][3], box['coordinates'][3])

            for layout_box in layout_data if layout_box['page'] == page
        )]
        non_overlapping_boxes.sort(key=lambda box: box['coordinates'][0])
        threshold = 50  # Increasing the threshold value will make the grouping criterion less stringent. This means that boxes can be farther apart and still be grouped together
        group = []
        for box in non_overlapping_boxes:
            if not group or abs(box['coordinates'][1] - group[-1]['coordinates'][1]) < threshold:
                group.append(box)
            else:
                sentence, coordinates = merge_group(group)
                add_new_layout_data(new_layout_data, "text", "pdfplumber", sentence, coordinates, page)
                group = [box]
        if group:
            sentence, coordinates = merge_group(group)
            add_new_layout_data(new_layout_data, "text", "pdfplumber", sentence, coordinates, page)
                
    return new_boxes, new_layout_data

def merge_group(group):
    sentence = ' '.join(b['content'] for b in group)
    coordinates = [
        min(b['coordinates'][0] for b in group),
        min(b['coordinates'][1] for b in group),
        max(b['coordinates'][2] for b in group),
        max(b['coordinates'][3] for b in group),
    ]
    return sentence, coordinates

def add_new_layout_data(new_layout_data, _type, source, content, coordinates, page):
    new_layout_data.append({
        "type": _type,
        "id": str(uuid.uuid4()),
        "source": source,
        "content": content,
        "coordinates": coordinates,
        "page": page,
    })

def generate_tree(new_layout_data):
    tree = []
    for page in set(box['page'] for box in new_layout_data):
        page_boxes = [box for box in new_layout_data if box['page'] == page]
        # Sort boxes by their y coordinate first, then by their x coordinate
        page_boxes.sort(key=lambda box: (box['coordinates'][1], box['coordinates'][0]))

        current_parent = None
        for box in page_boxes:
            node = {
                'type': box['type'],
                'id': box['id'],
                'source': box['source'],
                'content': box['content'],
                'coordinates': box['coordinates'],
                'page': box['page'],
                'children': [],
            }

            if box['type'] == 'Title' or current_parent is None:
                tree.append(node)
                current_parent = node
            else:
                current_parent['children'].append(node)

    return tree

