import json
import os
from PIL import Image, ImageEnhance

def delete_files(*file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def get_bbox(img_path, input_path):
    with open(os.path.join(input_path, "annotations.json"), "r") as f:
        data = json.load(f)

    img_id = os.path.basename(img_path)

    for item in data:
        if item["image_id"] == img_id:
            bbox = item["trench"]
            return bbox

    return None

def load_image(img_path, bbox, saturation=1.0, contrast=1.0, sharpness=1.0, cropped=False, black=False):

    image = Image.open(img_path).convert("RGB")

    if saturation != 1.0:
        color_enhancer = ImageEnhance.Color(image)
        image = color_enhancer.enhance(saturation)

    if contrast != 1.0:
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(contrast)

    if sharpness != 1.0:
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(sharpness)

    image_crop = image.crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))

    image_black = Image.new("RGB", image.size, (0, 0, 0))
    image_black.paste(image_crop, (bbox["xmin"], bbox["ymin"]))

    return image, image_crop, image_black