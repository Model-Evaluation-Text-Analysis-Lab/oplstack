import json

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

# Load the output of the LayoutParser
with open('preprocessing_pipeline/documents/lp_output_words.json', 'r') as f:
    lp_words_data = json.load(f)

# Load the output of pdfplumber
with open('preprocessing_pipeline/documents/pdfplumber_output.json', 'r') as f:
    pdfplumber_data = json.load(f)

iou_tolerance = 0.5  # define your tolerance here

# Iterate over the layout words data
for word_box in lp_words_data:
    # Initialize a content list for each box
    word_box['pdfplumber_content'] = []
    # Get the coordinates of the current layout word box
    layout_word_coords = word_box['coordinates']
    # Iterate over the pdfplumber data
    for word in pdfplumber_data:
        # Get the coordinates of the current pdfplumber word
        pdfplumber_word_coords = [word['x0'], word['top'], word['x1'], word['bottom']]
        # Calculate the IOU
        iou = bb_intersection_over_union(layout_word_coords, pdfplumber_word_coords)
        # If the IOU is greater than the defined tolerance, consider the words as matching
        if iou >= iou_tolerance:
            word_box['pdfplumber_content'].append(word['text'])

    # Join the words to form sentences
    word_box['pdfplumber_content'] = ' '.join(word_box['pdfplumber_content'])

# Save the merged data
with open('preprocessing_pipeline/documents/merged_output.json', 'w') as f:
    json.dump(lp_words_data, f, indent=4)

