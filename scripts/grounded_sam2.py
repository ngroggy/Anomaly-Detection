import argparse
import os
import cv2
import torch
import numpy as np
import supervision as sv
from supervision.draw.color import ColorPalette
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model
import groundingdino.datasets.transforms as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groundingdino.inference import get_grounding_output
from utils.general.dataloader import load_image, get_bbox, create_output_dir
from utils.groundedsam2.tools.helper_functions import filter_objects, filter_objects_with_full_image, generate_input, save_json_results

def main(args):

    # models
    groundingdino_model_cfg = args.groundingdino_model_config # "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    groundingdino_checkpoint = args.groundingdino_checkpoint # "gdino_checkpoints/groundingdino_swint_ogc.pth"
    sam2_checkpoint = args.sam2_checkpoint
    sam2_model_cfg = args.sam2_model_config

    # inputs
    threshold = args.box_threshold
    text_threshold = args.text_threshold
    saturation = args.saturation
    contrast = args.contrast
    sharpness = args.sharpness
    rerun = args.rerun
    input_dir = args.input_dir
    DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    # VERY important: text queries need to be lowercased + end with a dot
    prompt = args.text_prompt
    prompt_engineering = args.prompt_engineering

    filter_prompt = " "
    classes, prompt, token_spans = generate_input(prompt, filter_classes = [], filter_prompt=filter_prompt, prompt_engineering=prompt_engineering)
    print(prompt)
    if token_spans is not None:
        text_threshold = None

    phrase = ""
    for class_name in classes:
        if not (class_name in ["excavator shovel", "excavator arm"]):
            phrase += class_name + ', '

    print("\n\nLooking for: ", phrase[:-2] + ".", "\n\n")

    # outputs
    output_dir = create_output_dir(rerun, args.output_dir)

    # environment settings - use bfloat16
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
        model_config_path=groundingdino_model_cfg, 
        model_checkpoint_path=groundingdino_checkpoint,
        device=DEVICE
    )

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            
            img_path  = os.path.join(input_dir, filename)
            print(f"\nProcessing image: {filename}")

            bbox = get_bbox(img_path, input_dir)

            if bbox is None:
                print(f"No trench in {filename}, skipping.")
                continue

            image, image_crop, image_black = load_image(img_path, saturation=saturation, contrast=contrast, sharpness=sharpness, bbox=bbox, cropped=True, black=True)
            image_dict = {"full": image, "crop": image_crop, "black": image_black}

            full_img = image.copy()

            for key, image in image_dict.items():
                
                print(f"Processing image: {filename} - [{key}] \n")

                """
                Run Grounding-Dino Inference
                """
                transform = T.Compose(
                    [
                        T.RandomResize([800], max_size=1333),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

                image_transformed, _ = transform(image, None)
                
                boxes, labels = get_grounding_output(
                    grounding_model, image_transformed, prompt, threshold, text_threshold, cpu_only=False, token_spans=eval(f"{token_spans}")
                )

                # Process the box prompt for SAM 2
                h, w, _ = np.array(image).shape
                boxes = boxes * torch.Tensor([w, h, w, h])
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                # Segment the detected objects using SAM2 and save the output to the output directory
                if input_boxes.size == 0:
                    print(f"No objects detected, skipping {filename} [{key}] for SAM2.")

                    img = cv2.cvtColor(np.array(full_img), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, key, filename), img)

                    save_json_results(output_dir, None, None, None, None, image, img_path, key)

                else: 

                    """
                    Run SAM2 Inference
                    """
                    sam2_predictor.set_image(np.array(image))  

                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )

                    """
                    Post-process the output of the model to get the masks, scores, and logits for visualization
                    """

                    # # Remove detections of excavators 
                    # masks, input_boxes, labels = filter_objects(masks, input_boxes, labels, iou_threshold=0.8, exclude_keyword="excavator")

                    if isinstance(masks, list):
                        masks = np.array(masks)

                    # Handle empty masks
                    if masks.size == 0:
                        print(f"No object masks detected, skipping {filename} [{key}].")

                        img = cv2.cvtColor(np.array(full_img), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(output_dir, key, filename), img)

                        save_json_results(output_dir, None, None, None, None, image, img_path, key)
                        continue

                    # Convert the shape to (n, H, W)
                    if masks.ndim == 4:
                        masks = masks.squeeze(1)

                    """
                    Filter out detections on cropped image that intersect with excavator detections on the full image.
                    """ 

                    annotated_frame = cv2.cvtColor(np.array(full_img), cv2.COLOR_RGB2BGR)

                    if key == "crop":
                        # Adjust coordinates of masks and boxes for full image
                        x_min, y_min, x_max, y_max = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
                        adjusted_boxes = input_boxes.copy()
                        adjusted_boxes[:, [0, 2]] += x_min 
                        adjusted_boxes[:, [1, 3]] += y_min 

                        # Update mask positions to match full image coordinates
                        adjusted_masks = np.zeros((masks.shape[0], annotated_frame.shape[0], annotated_frame.shape[1]), dtype=bool)
                        for i, mask in enumerate(masks):
                            adjusted_masks[i, y_min:y_max, x_min:x_max] = cv2.resize(
                                mask.astype(np.uint8), (x_max - x_min, y_max - y_min)
                            ).astype(bool)
                    
                        masks, input_boxes, labels = filter_objects_with_full_image(
                                                        adjusted_masks, adjusted_boxes, labels,
                                                        full_image_json_path=os.path.join(output_dir, "full", "predictions.json"),
                                                        image_id = img_path,
                                                        iou_threshold=0.25,
                                                        exclude_keyword="excavator"
                                                    )
                        
                    """
                    Visualize image with supervision useful API
                    """ 

                    class_names = labels
                    class_ids = np.array(list(range(len(class_names))))

                    detections = sv.Detections(
                        xyxy=input_boxes,  # (n, 4)
                        mask=masks.astype(bool),  # (n, h, w)
                        class_id=class_ids
                    )
                    
                    box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
                    annotated_frame = box_annotator.annotate(scene=annotated_frame.copy(), detections=detections)

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
    parser.add_argument("--text-prompt", default="pipe. shovel. cable. tool. tube. single large stone. barrier.")
    parser.add_argument("--prompt-engineering", default=True)
    parser.add_argument("--token-spans", default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog' [2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog' [0:1] is 'a', and 'a cat and a dog' [2:5] is 'cat'. \
                        ")
    parser.add_argument("--input-dir", default="data/")
    parser.add_argument("--sam2-checkpoint", default="ckpts/grounded_sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs/GroundedSAM2/")
    parser.add_argument("--text-threshold",type=float, default=0.30)
    parser.add_argument("--box-threshold",type=float, default=0.30)
    parser.add_argument("--rerun", default=False)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    main(args)