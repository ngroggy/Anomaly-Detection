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
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model
import groundingdino.datasets.transforms as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groundingdino.inference import get_grounding_output
from utils.general.dataloader import load_image, get_bbox, create_output_dir


def print_classes(prompt, token_spans):
    token_spans_list = eval(f"{token_spans}")
    phrase = ''
    for token_span in token_spans_list:
        phrase += ' '.join([prompt[_s:_e] for (_s, _e) in token_span]) + ', '
    print("\n\nLooking for: ", phrase, "\n\n")


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
    GROUNDING_DINO_CONFIG = args.groundingdino_model_config #"grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = args.groundingdino_checkpoint #"gdino_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    sam2_checkpoint = args.sam2_checkpoint
    sam2_model_cfg = args.sam2_model_config

    # inputs
    # VERY important: text queries need to be lowercased + end with a dot
    prompt = args.text_prompt
    token_spans = args.token_spans
    if token_spans is not None:
        TEXT_THRESHOLD = None
        print_classes(prompt, token_spans)
    input_dir = args.input_dir
    DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    saturation = args.saturation
    contrast = args.contrast
    sharpness = args.sharpness
    rerun = args.rerun
    
    # outputs
    output_dir = create_output_dir(rerun, args.output_dir)

    # environment settings
    # use bfloat16
    torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            img_path  = os.path.join(input_dir, filename)
            print(f"\nProcessing image: {filename}")

            bbox = get_bbox(img_path, input_dir)

            image, image_crop, image_black = load_image(img_path, saturation=saturation, contrast=contrast, sharpness=sharpness, bbox=bbox, cropped=True, black=True)
            image_dict = {"full": image, "crop": image_crop, "black": image_black}
            
            for key, image in image_dict.items():
                print(f"Processing image: {filename} - [{key}] \n")

                transform = T.Compose(
                    [
                        T.RandomResize([800], max_size=1333),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

                image_transformed, _ = transform(image, None)
            
                sam2_predictor.set_image(np.array(image))        

                # run model
                boxes, labels = get_grounding_output(
                    grounding_model, image_transformed, prompt, BOX_THRESHOLD, TEXT_THRESHOLD, cpu_only=False, token_spans=eval(f"{token_spans}")
                )

                labels_copy = labels.copy()
                removed = 0
                for i, label in enumerate(labels):
                    if "excavator shovel" in label:
                        labels_copy.pop(i-removed)
                        boxes = torch.cat((boxes[:(i-removed), :], boxes[(i-removed)+1:, :]))
                        removed += 1

                labels = labels_copy

                # process the box prompt for SAM 2
                h, w, _ = np.array(image).shape
                boxes = boxes * torch.Tensor([w, h, w, h])
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


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
                    
                    # confidences = confidences.numpy().tolist()
                    class_names = labels

                    class_ids = np.array(list(range(len(class_names))))

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
    parser.add_argument("--groundingdino-model-config", default="configs/groundingdino/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--groundingdino-checkpoint", default="ckpts/grounding_dino/groundingdino_swint_ogc.pth")
    parser.add_argument("--text-prompt", default="pipe. shovel. cable. tool. tube. large stone. barrier. an excavator digging a trench.")
    parser.add_argument("--token-spans", default=None,help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    parser.add_argument("--input-dir", default="data/")
    parser.add_argument("--sam2-checkpoint", default="ckpts/grounded_sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs/GroundedSAM2/")
    parser.add_argument("--text-threshold",type=float, default=0.35)
    parser.add_argument("--box-threshold",type=float, default=0.35)
    parser.add_argument("--rerun", default=False)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    main(args)