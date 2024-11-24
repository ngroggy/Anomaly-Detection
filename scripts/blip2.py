from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import json
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general.dataloader import load_image, get_bbox, create_output_dir

def main(args):

    model_id = args.blip2_model
    input_dir = args.input_dir
    rerun = args.rerun
    saturation = args.saturation
    contrast = args.contrast
    sharpness = args.sharpness
    prompt = args.text_prompt
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto", load_in_8bit=True) 
    output_dir = create_output_dir(rerun, args.output_dir)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):

            img_path  = os.path.join(input_dir, filename)

            # Get the bounding box of the trench
            bbox = get_bbox(img_path, input_dir)
            # Load the image
            image, image_crop, image_black  = load_image(img_path, bbox, saturation=saturation, contrast=contrast, sharpness=sharpness)

            image_dict = {"full": image, "crop": image_crop, "black": image_black}
            print(f"Processing image: {filename}\n")

            for key, image in image_dict.items():
                print(f"Processing image: {filename} - [{key}] \n")

                # Prepare the output JSON file
                output_file = os.path.join(output_dir, key, 'predictions.json')
                
                # Load existing data if the output file already exists
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        output_data = json.load(f)
                else:
                    output_data = {}

                inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

                generated_ids = model.generate(**inputs, max_new_tokens=40)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                # Save the result to the output dictionary
                output_data[filename] = generated_text

                # Write to JSON after every prediction
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=4)
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blip2-model", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--text-prompt", default="This is a picture of a trench that has been dug by an excavator. The trench contains small rocks, dirt")
    parser.add_argument("--input-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/Blip2/")
    parser.add_argument("--rerun", default=False)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--force-cpu", default=False)
    args = parser.parse_args()
    main(args)