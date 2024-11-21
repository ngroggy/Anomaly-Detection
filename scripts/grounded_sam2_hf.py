import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from PIL import Image, ImageEnhance
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

def get_bbox(img_path, input_path):
    with open(os.path.join(input_path, "annotations.json"), "r") as f:
        data = json.load(f)

    img_id = os.path.basename(img_path)

    for item in data:
        if item["image_id"] == img_id:
            bbox = item["trench"]
            return bbox

    return None


def load_image(img_path, bbox, saturation=1.0, contrast=1.0, cropped=False, black=False):

    image = Image.open(img_path).convert("RGB")

    if saturation != 1.0:
        color_enhancer = ImageEnhance.Color(image)
        image = color_enhancer.enhance(saturation)

    if contrast != 1.0:
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(contrast)

    image_crop = image.crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))

    image_black = Image.new("RGB", image.size, (0, 0, 0))
    image_black.paste(image_crop, (bbox["xmin"], bbox["ymin"]))

    return image, image_crop, image_black


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


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


def main(args):

    # models
    grounding_model_id = args.grounding_model
    sam2_checkpoint = args.sam2_checkpoint
    sam2_model_cfg = args.sam2_model_config

    # inputs
    # VERY important: text queries need to be lowercased + end with a dot
    prompt = args.text_prompt
    input_dir = args.input_dir
    DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    saturation = args.saturation
    contrast = args.contrast
    
    # outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_length = len(os.listdir(output_dir))
    rerun = args.rerun

    if rerun is True:
        output_dir = Path(os.path.join(output_dir, f"{output_dir_length:02d}"))
    else:
        output_dir = Path(os.path.join(output_dir, f"{output_dir_length + 1:02d}"))

    # output_dir.mkdir(parents=True, exist_ok=True)

    for key in ["full", "crop", "black"]:
        output_dir_key = os.path.join(output_dir, key)
        os.makedirs(output_dir_key, exist_ok=True)

    # environment settings
    # use bfloat16
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino from huggingface
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(DEVICE)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            img_path  = os.path.join(input_dir, filename)
            print(f"\nProcessing image: {filename}")
            # image = Image.open(img_path)

            bbox = get_bbox(img_path, input_dir)

            image, image_crop, image_black = load_image(img_path, saturation=saturation, contrast=contrast, bbox=bbox, cropped=True, black=True)
            image_dict = {"full": image, "crop": image_crop, "black": image_black}
            
            for key, image in image_dict.items():

                if image is None:
                    continue
            
                sam2_predictor.set_image(np.array(image))        

                inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = grounding_model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.2,
                    text_threshold=0.2,
                    target_sizes=[image.size[::-1]]
                )

                """
                Results is a list of dict with the following structure:
                [
                    {
                        'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
                        'labels': ['car', 'tire', 'tire', 'tire'], 
                        'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                                        [1392.4701,  554.4064, 1628.6133,  777.5872],
                                        [ 436.1182,  621.8940,  676.5255,  851.6897],
                                        [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
                    }
                ]
                """

                # get the box prompt for SAM 2
                input_boxes = results[0]["boxes"].cpu().numpy()

                # Segment the detected objects using SAM2 and save the output to the output directory
                if input_boxes.size == 0:
                    print(f"No objects detected, skipping {filename} [{key}] for SAM2.")

                    # img = cv2.imread(img_path)
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, key, filename), img)

                    save_json_results(output_dir, None, None, None, None, image, img_path, key)

                else: 
                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )


                    """
                    Post-process the output of the model to get the masks, scores, and logits for visualization
                    """
                    # convert the shape to (n, H, W)
                    if masks.ndim == 4:
                        masks = masks.squeeze(1)


                    confidences = results[0]["scores"].cpu().numpy().tolist()
                    class_names = results[0]["labels"]
                    class_ids = np.array(list(range(len(class_names))))

                    labels = [
                        f"{class_name} {confidence:.2f}"
                        for class_name, confidence
                        in zip(class_names, confidences)
                    ]

                    """
                    Visualize image with supervision useful API
                    """
                    # img = cv2.imread(img_path)
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    detections = sv.Detections(
                        xyxy=input_boxes,  # (n, 4)
                        mask=masks.astype(bool),  # (n, h, w)
                        class_id=class_ids
                    )

                    box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
                    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

                    label_annotator = sv.LabelAnnotator(color=ColorPalette.DEFAULT)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

                    mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)
                    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                    cv2.imwrite(os.path.join(output_dir, key, filename), annotated_frame)


                    """
                    Dump the results in standard format and save as json files
                    """

                    save_json_results(output_dir, masks, scores, class_names, input_boxes, image, img_path, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--text-prompt", default="pipe. shovel. cable. tool. tube. boulder. barrier.")
    parser.add_argument("--input-dir", default="data/")
    parser.add_argument("--sam2-checkpoint", default="ckpts/grounded_sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs/GroundedSAM2/")
    parser.add_argument("--rerun", default=False)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    main(args)