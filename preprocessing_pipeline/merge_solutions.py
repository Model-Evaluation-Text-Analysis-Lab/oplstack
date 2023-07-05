#%%
import json
import uuid

#%%
# Function to compute Intersection over Union
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

#%%
# Load the output of the LayoutParser
with open('preprocessing_pipeline/output_files/PDF/lp_output_words.json', 'r') as f:
    lp_words_data = json.load(f)

# Load the output of pdfplumber
with open('preprocessing_pipeline/output_files/PDF/pdfplumber_output.json', 'r') as f:
    pdfplumber_data = json.load(f)

#%%
iou_tolerance = 0.1  # define your tolerance here
pdfplumber_data_matched = set()
lp_words_data_extended = lp_words_data[:]
# Iterate over the layout words data
for word_box in lp_words_data:
    # Initialize a content list for each box
    word_box['pdfplumber_content'] = []
    # Get the coordinates of the current layout word box
    layout_word_coords = word_box['coordinates']
    # Iterate over the pdfplumber data
    for idx, word in enumerate(pdfplumber_data):
        pdfplumber_word_coords = [word['x0'], word['top'], word['x1'], word['bottom']]
        iou = bb_intersection_over_union(layout_word_coords, pdfplumber_word_coords)

        if iou >= iou_tolerance:
            word_box['pdfplumber_content'].append(word['text'])
            pdfplumber_data_matched.add(idx)

    word_box['pdfplumber_content'] = ' '.join(word_box['pdfplumber_content'])
    
# Append unmatched pdfplumber data as text with new uuid
for idx, word in enumerate(pdfplumber_data):
    if idx not in pdfplumber_data_matched:
        print(word['text'])
        new_uuid = str(uuid.uuid4())
        new_word_box = {
            'uuid': new_uuid,
            'type': 'text',
            'content': "",
            'coordinates': [word['x0'], word['top'], word['x1'], word['bottom']],
            'pdfplumber_content': word['text']
        }
        lp_words_data_extended.append(new_word_box)


#%%
# Save the merged data
with open('preprocessing_pipeline/output_files/PDF/merged_output.json', 'w') as f:
    json.dump(lp_words_data_extended, f, indent=4)
    
# %%
