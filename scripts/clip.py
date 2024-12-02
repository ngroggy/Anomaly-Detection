from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import json
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general.dataloader import load_image, get_bbox, create_output_dir

trench_templates = [
    'a photo of the {} laying in a trench.',
    'a photo of a {} laying in a trench.',
    'a photo of the {} covered in dirt.',
    'a photo of a {} covered in dirt.',
    'a photo of the {} buried in dirt.',
    'a photo of a {} buried in dirt.',
    'a photo of the {} sticking out of dirt.',
    'a photo of a {} sticking out of dirt.',
    # 'itap of a {}.',
    # 'a bad photo of the {}.',
    # 'a origami {}.',
    # 'a photo of the large {}.',
    # 'a {} in a video game.',
    # 'art of the {}.',
    # 'a photo of the small {}.'
]

def create_text_descriptions(prompt, prompt_engineering=False):
    classes = [desc.strip() for desc in prompt.split('.') if desc.strip()]
    classes_engineered = []
    if prompt_engineering:
        for classname in classes:
            classes_engineered.extend([template.format(classname) for template in trench_templates])
    return classes, classes_engineered

def main(args):

    model_id = args.clip_model
    input_dir = args.input_dir
    rerun = args.rerun
    threshold= args.threshold
    saturation = args.saturation
    contrast = args.contrast
    sharpness = args.sharpness
    prompt = args.text_prompt
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    prompt_engineering = args.prompt_engineering

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    output_dir = create_output_dir(rerun, args.output_dir)

    classes, text_descriptions = create_text_descriptions(prompt, prompt_engineering)

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

                inputs = processor(
                    text=text_descriptions,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(device)

                with torch.no_grad():
                    image_features = model.get_image_features(inputs["pixel_values"])
                    text_features = model.get_text_features(inputs["input_ids"])

                if prompt_engineering:
                    zeroshot_weights = []
                    for i in range(len(classes)):
                        class_embeddings=text_features[i*len(trench_templates):(i+1)*len(trench_templates)]
                        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        zeroshot_weights.append(class_embedding)
                    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
                else:
                    zeroshot_weights = (text_features / text_features.norm(dim=-1, keepdim=True)).T

                # Normalize the embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarities =  image_features @ zeroshot_weights
                similarity_scores = similarities.squeeze(0).cpu().numpy()

                generated_text = []

                for object, prob in zip(classes, similarity_scores):
                    if prob >= threshold and not ("excavator" in object):
                        generated_text.extend([f"{object}: {100*prob:.2f}%"])

                # Save the result to the output dictionary
                output_data[filename] = generated_text

                # Write to JSON after every prediction
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--text-prompt", default="pipe. shovel. excavator shovel. cable. tool. wire. tube. single large rock. construction barrier. excavator.")
    parser.add_argument("--prompt-engineering", default=True)
    parser.add_argument("--input-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/Clip/")
    parser.add_argument("--rerun", default=False)
    parser.add_argument("--threshold",type=float, default=0.3)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--force-cpu", default=False)
    args = parser.parse_args()
    main(args)
