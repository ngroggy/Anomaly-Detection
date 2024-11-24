from PIL import Image
import torch
import numpy as np
import random
import json
import os
import argparse
from pycocotools import mask
from transformers import AutoProcessor
from transformers import AutoModelForUniversalSegmentation
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general.dataloader import load_image, get_bbox, create_output_dir

def annotate_image(image, segmentation):
   # Convert segmentation tensor to a NumPy array
    segmentation_np = segmentation.numpy()

    # Generate a unique color for each segment ID
    unique_ids = np.unique(segmentation_np)
    colors = {id_: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for id_ in unique_ids}

    # Create a blank RGB mask image
    segmentation_colored = np.zeros((*segmentation_np.shape, 3), dtype=np.uint8)

    # Apply the colors to the segmentation
    for seg_id, color in colors.items():
        segmentation_colored[segmentation_np == seg_id] = color

    # Convert the mask to a PIL image
    segmentation_colored_img = Image.fromarray(segmentation_colored)

    # Resize the segmentation mask to match the original image size
    segmentation_colored_img = segmentation_colored_img.resize(image.size, resample=Image.NEAREST)

    # Overlay the segmentation mask on the original image
    overlay_image = Image.blend(image, segmentation_colored_img, alpha=0.6)

    return overlay_image

def main(args):
    model_id = args.oneformer_model
    input_dir = args.input_dir
    rerun = args.rerun
    saturation = args.saturation
    contrast = args.contrast
    sharpness = args.sharpness
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    output_dir = create_output_dir(rerun, args.output_dir)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_id)
    model.to(device)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):

            img_path  = os.path.join(input_dir, filename)
            print(f"\nProcessing image: {filename}")

            bbox = get_bbox(img_path, input_dir)

            image, image_crop, image_black = load_image(img_path, saturation=saturation, contrast=contrast, sharpness=sharpness, bbox=bbox, cropped=True, black=True)
            image_dict = {"full": image, "crop": image_crop, "black": image_black}

            for key, img in image_dict.items():
                print(f"\nProcessing image: {filename} - [{key}]")

                ## Instance Segmentation
                # prepare image for the model
                instance_inputs = processor(images=img, task_inputs=["panoptic"], return_tensors="pt").to(device)

                # forward pass
                with torch.no_grad():
                    outputs = model(**instance_inputs)

                instance_segmentation = processor.post_process_instance_segmentation(outputs)[0]

                if instance_segmentation["segments_info"]: 
                    image_annotated = annotate_image(img, instance_segmentation["segmentation"].cpu())
                else:
                    image_annotated = img

                annotations = {
                        "class_names": [],
                        "scores": [],
                        "segmentation": [
                            {
                                "size": [],
                                "counts": []
                            }
                        ]
                    }
                for segment in instance_segmentation["segments_info"]:
                    segment_label_id = segment['label_id']
                    label = model.config.id2label[segment_label_id]
                    score = segment['score']
                    
                    # Extract the binary mask for the current segment
                    mask_tensor = instance_segmentation["segmentation"] == segment["id"]
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    
                    # Encode segmentation mask into RLE
                    rle = mask.encode(np.asfortranarray(mask_np))
                    rle["counts"] = rle["counts"].decode("utf-8")  # Convert bytes to string for JSON
                    
                    annotations["class_names"].append(label)
                    annotations["scores"].append(score)
                    annotations["segmentation"][0]["size"].append(list(rle["size"]))
                    annotations["segmentation"][0]["counts"].append(rle["counts"])

                image_annotated.save(os.path.join(output_dir, key, filename))

                # Build the final output dictionary
                output_data = {
                    "image_path": filename,
                    "annotations": annotations,
                    "img_width": img.width,
                    "img_height": img.height
                }

                # Load existing data or initialize an empty list
                predictions_path = os.path.join(output_dir, key, "predictions.json")
                existing_results = []
                if os.path.exists(predictions_path):
                    with open(predictions_path, "r") as f:
                        existing_results = json.load(f)
                
                # Append new result
                existing_results.append(output_data)
                
                # Write back to the file
                with open(predictions_path, "w") as f:
                    json.dump(existing_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oneformer-model", default="shi-labs/oneformer_ade20k_dinat_large")
    parser.add_argument("--input-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/OneFormer/")
    parser.add_argument("--rerun", default=False)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--force-cpu", default=False)
    args = parser.parse_args()
    main(args)