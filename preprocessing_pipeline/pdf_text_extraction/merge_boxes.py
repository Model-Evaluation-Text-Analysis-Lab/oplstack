import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import uuid


def merge_boxes(all_word_data):
    boxes_by_page = {}
    for box in all_word_data:
        page = box['page']
        if page not in boxes_by_page:
            boxes_by_page[page] = []
        boxes_by_page[page].append(box)

    new_boxes = []

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
            y_min = max(coordinate[1] for coordinate in coordinates)
            x_max = max(coordinate[2] for coordinate in coordinates)
            y_max = min(coordinate[3] for coordinate in coordinates)
            union_coordinates = [x_min, y_min, x_max, y_max]

            new_boxes.append({
                'uid': str(uuid.uuid4()),
                'type': 'Word',
                'content': content.strip(),
                'discarded_content': discarded_content.strip(),
                'coordinates': union_coordinates,
                'page': page
            })

    return new_boxes
