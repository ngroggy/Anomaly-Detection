import json
import numpy as np
import cv2
import pycocotools.mask as mask_util

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
    to_keep = set(range(num_masks))  # Indices of masks to keep

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

    # Filter detections by image_id
    image_detections = [det for det in full_image_detections if det.get("image_path") == image_id]
    if not image_detections:
        raise ValueError(f"No detections found for image_id: {image_id}")
    
    # Initialize lists for masks and boxes from the full image
    full_image_masks = []
    full_image_boxes = []

    annotations = image_detections[0]["annotations"]  # Access the annotations list
    
    # Iterate over each annotation entry
    for annotation in annotations:
        for class_name, bbox, seg in zip(annotation["class_names"], annotation["bbox"], annotation["segmentation"]):
            if "excavator" in class_name:  # Check if the label contains "excavator"
                # Decode RLE to binary mask if segmentation is in RLE format
                if isinstance(seg, dict):  # RLE format
                    binary_mask = mask_util.decode(seg)
                else:
                    print(f"Skipping unsupported segmentation format: {seg}")
                    continue

                # Append decoded masks and corresponding bounding boxes
                full_image_masks.append(binary_mask)
                full_image_boxes.append(bbox)

    # Convert full image boxes to NumPy array
    full_image_boxes = np.array(full_image_boxes) if full_image_boxes else np.empty((0, 4))

    # Initialize set to track indices of masks to keep
    num_masks = len(masks)
    to_keep = set(range(num_masks))

    # Filter masks based on intersection over object masks
    for i in range(num_masks):
        for full_mask in full_image_masks:
            # Compute intersection and the area of the object mask
            intersection = np.logical_and(masks[i], full_mask).sum()
            object_area = masks[i].sum()  # Area of the object mask

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