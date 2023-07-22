def generate_tree(new_layout_data):
    tree = {}

    for page in set(box['page'] for box in new_layout_data):
        tree[page] = []
        page_boxes = [box for box in new_layout_data if box['page'] == page]
        sorted_boxes = sorted(page_boxes, key=lambda x: x['reading_order'])

        parent = None

        for box in sorted_boxes:
            node = {
                'type': box['type'],
                'id': box['id'],
                'source': box['source'],
                'content': box['content'],
                'coordinates': box['coordinates'],
                'page': box['page'],
                'children': [],
                'size': box.get('size', None),
            }

            if node['type'] == 'Title' or parent is None:
                parent = node
                tree[page].append(parent)
            else:
                parent['children'].append(node)

    return tree
