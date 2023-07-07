#%%
import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import uuid
#%%
# Load the output of the LayoutParser
with open('preprocessing_pipeline/output_files/PDF/lp_output_words.json', 'r') as f:
    lp_words_data = json.load(f)

# Load the output of pdfplumber
with open('preprocessing_pipeline/output_files/PDF/pdfplumber_output.json', 'r') as f:
    pdfplumber_data = json.load(f)
# %%
all_boxes = lp_words_data + pdfplumber_data

# Convert all_boxes to a numpy array
all_boxes_np = np.array([box['coordinates'] for box in all_boxes])
all_boxes_idx = {idx: box['uid'] for idx, box in enumerate(all_boxes)}

x_max = np.maximum(all_boxes_np[:, None, 0], all_boxes_np[:, 0])  # xA in IoU
y_max = np.maximum(all_boxes_np[:, None, 1], all_boxes_np[:, 1])  # yA in IoU
x_min = np.minimum(all_boxes_np[:, None, 2], all_boxes_np[:, 2])  # xB in IoU
y_min = np.minimum(all_boxes_np[:, None, 3], all_boxes_np[:, 3])  # yB in IoU

# Compute areas
interArea = np.maximum(0, x_min - x_max + 1) * np.maximum(0, y_min - y_max + 1)
boxAArea = (all_boxes_np[:, 2] - all_boxes_np[:, 0] + 1) * (all_boxes_np[:, 3] - all_boxes_np[:, 1] + 1)
boxBArea = (all_boxes_np[:, None, 2] - all_boxes_np[:, None, 0] + 1) * (all_boxes_np[:, None, 3] - all_boxes_np[:, None, 1] + 1)
iou = interArea / (boxAArea + boxBArea - interArea)

# Create adjacency matrix
graph = (iou >= 0.1).astype(int)
# Convert to CSR sparse matrix representation
graph_csr = csr_matrix(graph)

# Find the connected components
n_components, labels = connected_components(csgraph=graph_csr, directed=False, return_labels=True)
        
#%%    
new_boxes = []

# Loop over each connected component
for label in range(n_components):
    # Get the indices of the boxes in this connected component
    indices = np.where(labels == label)[0]

    # Initialize an empty string to store the concatenated contents and a list to store the coordinates
    content = ''
    discarded_content = ''
    coordinates = []

    # Loop over the boxes in this connected component
    for idx in indices:
        box = all_boxes[idx]
        # Concatenate the content
        if box['source'] == 'pdfplumber':
            content += ' ' + box['content']
        else:
            discarded_content += ' ' + box['content']

        # Add the coordinates to the list
        coordinates.append(box['coordinates'])

    # Take the union of the coordinates
    x_min = min(coordinate[0] for coordinate in coordinates)
    y_min = min(coordinate[1] for coordinate in coordinates)
    x_max = max(coordinate[2] for coordinate in coordinates)
    y_max = max(coordinate[3] for coordinate in coordinates)
    union_coordinates = [x_min, y_min, x_max, y_max]

    # Add the new box to the list
    new_boxes.append({
        'uid': str(uuid.uuid4()),
        'type': 'Word',
        'content': content.strip(),
        'discarded_content': discarded_content.strip(),
        'coordinates': union_coordinates,
    })

# Save the new boxes to a JSON file
with open('preprocessing_pipeline/output_files/PDF/merged_output.json', 'w') as f:
    json.dump(new_boxes, f, indent=4)

# %%
