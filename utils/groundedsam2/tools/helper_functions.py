import json
import numpy as np
import re
import pycocotools.mask as mask_util
import os


def print_classes(prompt, token_spans):
    token_spans_list = eval(f"{token_spans}")
    phrase = ''
    for token_span in token_spans_list:
        phrase += ' '.join([prompt[_s:_e] for (_s, _e) in token_span]) + ', '
    print("\n\nLooking for: ", phrase, "\n\n")

def extractClassNames(prompt):
    classes = []
    for desc in prompt.split('.'):
        if desc.strip():
            classes.append(desc.strip())

    return classes

def createTextPrompts(classes, trench_templates = ["a {} laying in a trench."]):
    classes_engineered = []
    for classname in classes:
        classes_engineered.extend([template.format(classname) for template in trench_templates])
    return classes_engineered

def get_token_positions(text, classes):
    """
    Finds the start and end positions of all occurrences of class words or phrases in the text,
    ensuring that overlapping matches are not reused for different classes.
    Longer classes are prioritized over shorter ones.

    Args:
        text (str): The input text to search.
        classes (list): A list of class words or phrases to find.

    Returns:
        list: A concatenated 2-level nested list containing spans for all classes.
    """
    all_spans = []  # This will hold the concatenated spans for all classes
    used_positions = set()  # To keep track of positions already matched

    # Sort classes by length (longest first) to prioritize longer matches
    classes = sorted(classes, key=len, reverse=True)

    for cls in classes:
        # Use regex to find all matches of the class in the text
        for match in re.finditer(re.escape(cls), text):
            words = cls.split()  # Handle multi-word classes
            span = []
            start = match.start()

            valid_match = True  # Flag to check if this match overlaps with used positions
            for word in words:
                word_start = text.find(word, start)
                word_end = word_start + len(word)

                # Check if any position in this word range has already been used
                if any(pos in used_positions for pos in range(word_start, word_end)):
                    valid_match = False
                    break

                span.append([word_start, word_end])
                start = word_end  # Update start for next word in multi-word class

            if valid_match:
                all_spans.append(span)
                # Mark these positions as used
                for word_start, word_end in span:
                    used_positions.update(range(word_start, word_end))
    
    return all_spans


def generate_input(prompt, filter_prompt, filter_classes):
        classes = extractClassNames(prompt)
        prompt = " ".join(createTextPrompts(classes)) + filter_prompt
        classes.extend(filter_classes)
        tokespans = get_token_positions(prompt, classes)
        return classes, prompt, tokespans

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def filter_objects(masks, boxes, labels, iou_threshold=0.8, exclude_keyword="excavator"):
    """
    Remove objects whose masks are completely or almost fully inside another mask and
    exclude objects with labels containing a specific keyword.
    
    Args:
        masks (list of np.ndarray): Binary masks of shape (H, W) for each object.
        boxes (np.ndarray): Corresponding bounding boxes of shape (N, 4) in xyxy format.
        labels (list of str): Labels corresponding to each object.
        iou_threshold (float): Threshold to determine if one mask is inside another.
        exclude_keyword (str): Keyword for labels to exclude (case-insensitive).
    
    Returns:
        filtered_masks (list of np.ndarray): Filtered binary masks.
        filtered_boxes (np.ndarray): Filtered bounding boxes.
        filtered_labels (list of str): Filtered labels.
    """
    num_masks = len(masks)

    # Indices of masks to keep
    to_keep = set(range(num_masks))  

    for i in range(num_masks):
        for j in range(num_masks):
            if i != j and i in to_keep and j in to_keep:
                # Compute IoU between masks[i] and masks[j]
                intersection = np.logical_and(masks[i], masks[j]).sum()
                area_i = masks[i].sum()
                iou = intersection / area_i

                # If mask i is almost fully inside mask j, remove mask i
                if iou > iou_threshold:
                    to_keep.remove(i)
                    break

    # Filter out objects with the exclude_keyword in their label
    to_keep = {i for i in to_keep if exclude_keyword.lower() not in labels[i].lower()}

    # Filter masks, boxes, and labels based on indices to keep
    filtered_masks = [masks[i] for i in to_keep]
    filtered_boxes = boxes[list(to_keep), :]
    filtered_labels = [labels[i] for i in to_keep]

    return filtered_masks, filtered_boxes, filtered_labels

def filter_objects_with_full_image(masks, boxes, labels, full_image_json_path, image_id, iou_threshold=0.8, exclude_keyword="excavator"):
    # Load full image detections from JSON file
    with open(full_image_json_path, "r") as f:
        full_image_detections = json.load(f)

    image_detections = [det for det in full_image_detections if det.get("image_path") == image_id]
    if not image_detections:
        raise ValueError(f"No detections found for image_id: {image_id}")
    
    # Retrieve bboxes and segmentation masks of excavator detections in the full image
    full_image_masks = []
    full_image_boxes = []

    annotations = image_detections[0]["annotations"]
    for annotation in annotations:
        for class_name, bbox, seg in zip(annotation["class_names"], annotation["bbox"], annotation["segmentation"]):
            if "excavator" in class_name:
                # Decode RLE to binary mask if segmentation is in RLE format
                if isinstance(seg, dict):
                    binary_mask = mask_util.decode(seg)
                else:
                    print(f"Skipping unsupported segmentation format: {seg}")
                    continue

                full_image_masks.append(binary_mask)
                full_image_boxes.append(bbox)

    full_image_boxes = np.array(full_image_boxes) if full_image_boxes else np.empty((0, 4))

    # Initialize set to track indices of masks to keep from the cropped image
    num_masks = len(masks)
    to_keep = set(range(num_masks))

    # Filter out every mask that intersects the excavator mask depending on threshold
    for i in range(num_masks):
        for full_mask in full_image_masks:
            # Compute intersection and the area of the object mask
            intersection = np.logical_and(masks[i], full_mask).sum()
            object_area = masks[i].sum() 

            # Calculate Intersection over Object Area
            ioa = intersection / object_area if object_area > 0 else 0

            if ioa > iou_threshold:
                to_keep.discard(i)
                break

    # Exclude objects based on the exclude_keyword in their labels
    to_keep = {i for i in to_keep if exclude_keyword.lower() not in labels[i].lower()}

    # Filter masks, boxes, and labels based on indices to keep
    filtered_masks = np.array([masks[i] for i in to_keep])
    # print(np.empty((0, masks[0].shape[0], masks[0].shape[1])).shape)
    if filtered_masks.size == 0:
        filtered_masks = np.empty((0, masks[0].shape[0], masks[0].shape[1]))
    filtered_boxes = np.array([boxes[i] for i in to_keep])
    if filtered_boxes.size == 0:
        filtered_boxes = np.empty((0, 4))
    filtered_labels = [labels[i] for i in to_keep]

    return filtered_masks, filtered_boxes, filtered_labels

def encode_polygon_to_rle(polygon):
    """
    Converts a polygon (list of points) into RLE format.
    You might need to use a utility function to convert the polygon into RLE format.
    """
    # Convert the polygon to RLE format (pycocotools has methods for this)
    rle = mask_util.frPyObjects(polygon, 260, 231)  # 260 and 231 are the height and width of the image
    return rle["counts"]

def save_json_results(output_dir, masks, scores, class_names, input_boxes, image, img_path, key):
    predictions_path = os.path.join(output_dir, key, "predictions.json")
    if masks is None:
        results = {
        "image_path": img_path,
        "annotations" : [],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }

    else:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": img_path,
            "annotations" : [
                {
                    "class_names": class_names,
                    "score": scores,
                    "bbox": input_boxes,
                    "segmentation": mask_rles,
                }
            ],
            "box_format": "xyxy",
            "img_width": image.width,
            "img_height": image.height,
        }
        
    # Load existing data or initialize an empty list
    existing_results = []
    if os.path.exists(predictions_path):
        with open(predictions_path, "r") as f:
            existing_results = json.load(f)
    
    # Append new result
    existing_results.append(results)
    
    # Write back to the file
    with open(predictions_path, "w") as f:
        json.dump(existing_results, f, indent=4)