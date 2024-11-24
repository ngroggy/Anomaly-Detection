import torch
from ultralytics import YOLO
import shutil
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general.dataloader import load_image, get_bbox, delete_files, create_output_dir


def structure_output(output_dir, filename):
    # Append detections to predictions.txt
    predictions_file = os.path.join(output_dir, "predictions.txt")
    predict_folder = os.path.join(output_dir, "predict")
    labels_folder = os.path.join(predict_folder, "labels")

    if not os.path.exists(predictions_file):
        open(predictions_file, 'w').close()  # Create file if it doesn't exist

    for txt_file in os.listdir(labels_folder):
        txt_path = os.path.join(labels_folder, txt_file)
        with open(txt_path, 'r') as f:
            detections = f.read()

        # Append to predictions.txt
        with open(predictions_file, 'a') as predictions:
            predictions.write(f"Image: {filename}\n")
            predictions.write(detections)
            predictions.write("\n")  # Add spacing between entries

    # Move all images from predict folder to project folder
    for img_file in os.listdir(predict_folder):
        if img_file.endswith(".jpg"):  # Only move image files
            shutil.move(os.path.join(predict_folder, img_file), os.path.join(output_dir, filename))

    # Remove the predict folder
    shutil.rmtree(predict_folder)


def main(args):
    model_id = args.yolo11_model
    input_dir = args.input_dir
    rerun = args.rerun
    saturation = args.saturation
    contrast = args.contrast
    sharpness = args.sharpness
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    output_dir = create_output_dir(rerun, args.output_dir)

    model = YOLO(model_id)

    # Process each .png image in the folder
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):

            img_path  = os.path.join(input_dir, filename)
            print(f"\nProcessing image: {filename}")

            bbox = get_bbox(img_path, input_dir)

            image, image_crop, image_black = load_image(img_path, saturation=saturation, contrast=contrast, sharpness=sharpness, bbox=bbox, cropped=True, black=True)
            image_dict = {"full": image, "crop": image_crop, "black": image_black}

            for key, img in image_dict.items():
                print(f"\nProcessing image: {filename} - [{key}]")
                model.predict(img, device=device, save=True, save_txt=True, save_conf=True, project=os.path.join(output_dir, key))
                structure_output(os.path.join(output_dir, key), filename)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo11-model", default="yolo11x.pt")
    parser.add_argument("--input-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/Yolo11/")
    parser.add_argument("--rerun", default=False)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--force-cpu", default=False)
    args = parser.parse_args()
    main(args)