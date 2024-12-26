import os
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time
import argparse
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general.dataloader import load_image, get_bbox, delete_files
from pathlib import Path

def process_images(input_folder, filename, output_folder, saturation=1.0, contrast=1.0, sharpness=1.0):

    img_path = os.path.join(input_folder, filename)

    # Get the bounding box of the trench
    bbox = get_bbox(img_path, input_folder)

    if bbox is None:
        print(f"No trench in {img_path}, skipping.")
        image, _,_ = load_image(img_path, bbox, saturation=saturation, contrast=contrast, sharpness=sharpness)
        image_path = os.path.join(output_folder, "full", filename)
        cv2.imwrite(image_path, np.array(image))
        return image_path, None, None

    # Load the image
    image, image_crop, image_black = load_image(img_path, bbox, saturation=saturation, contrast=contrast, sharpness=sharpness)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_crop = cv2.cvtColor(np.array(image_crop), cv2.COLOR_RGB2BGR)
    image_black = cv2.cvtColor(np.array(image_black), cv2.COLOR_RGB2BGR)

    image_path = os.path.join(output_folder, "full", filename)
    image_crop_path = os.path.join(output_folder, "crop", filename)
    image_black_path = os.path.join(output_folder, "black", filename)

    cv2.imwrite(image_path, image)
    cv2.imwrite(image_black_path, image_black)
    cv2.imwrite(image_crop_path, image_crop)

    return image_path, image_crop_path, image_black_path

def main(args):

    model_id = args.qwen2_model
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    device_map = "auto" if device == "cuda" else "cpu"
    prompt = args.text_prompt
    saturation = args.saturation
    contrast = args.contrast
    sharpness = args.sharpness

    # Folder containing the images
    input_folder = args.input_dir
    output_folder = args.output_dir

    # outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_length = len(os.listdir(output_dir))
    rerun = args.rerun

    if rerun is True:
        output_folder = Path(os.path.join(output_dir, f"{output_dir_length:02d}"))
    else:
        output_folder = Path(os.path.join(output_dir, f"{output_dir_length + 1:02d}"))

    for key in ["full", "crop", "black"]:
        os.makedirs(os.path.join(output_folder, key), exist_ok=True)

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Load the model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id ,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )

    # # Load the model on the available device(s) without flash_attention
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_id , torch_dtype=torch.bfloat16 , device_map=device_map
    # )

    model.to(device)

    # Process each .png image in the folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.png'):
            # if filename in output_data:
            #     print(f"Skipping already processed image: {filename}")
            #     continue
            
            image_path, image_crop_path, image_black_path = process_images(input_folder, 
                                                                           filename, 
                                                                           output_folder, 
                                                                           saturation=saturation, 
                                                                           contrast=contrast, 
                                                                           sharpness=sharpness)
            
            if image_black_path is None and image_crop_path is None:
                continue

            image_dict = {"full": image_path, "crop": image_crop_path, "black": image_black_path}
            print(f"Processing image: {filename}\n")

            for key, img_path in image_dict.items():

                # Prepare the output JSON file
                output_file = os.path.join(output_folder, key, 'predictions.json')
                
                # Load existing data if the output file already exists
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        output_data = json.load(f)
                else:
                    output_data = {}

                # Prepare the messages for the model
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_path,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                
                # Preparation for inference
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs = inputs.to(device)

                # Inference: Generation of the output
                # start = time.time()
                generated_ids = model.generate(**inputs, max_new_tokens=20)

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # print(f"Finished! Time taken for {filename}: ", time.time() - start, "seconds")
                print(f"Output {filename} - [{key}]:\n", output_text[0], "\n")

                # Save the result to the output dictionary
                output_data[filename] = output_text[0]

                # Write to JSON after every prediction
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=4)
                
            delete_files(image_path, image_crop_path, image_black_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen2-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--text-prompt", default="Detect the anomalies in the trench.")
    parser.add_argument("--input-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/Qwen2VL/")
    parser.add_argument("--rerun", default=False)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--force-cpu", default=False)
    args = parser.parse_args()
    main(args)