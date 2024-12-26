# Anomaly Detection

This repository is complementary to the semester thesis "Vision-Based Anomaly Detection for Autonomous Excavators" by Nic Grogg. 

This thesis focuses on visual anomaly detection during trench excavation by autonomous excavators, a critical aspect for safety and prevention of damage to excavators and existing infrastructure. Models from three categories, namely closed-vocabulary object detection, open-vocabulary object detection and visual language models, are evaluated. Two new datasets, an image dataset and a video dataset, are created from recordings of an excavator digging a trench and revealing anomalies, such as objects lying or buried in the trench, simulating real excavation scenarios with different types of anomalies. The image dataset consists of single frames from these recordings, while the video dataset contains several videos of digging movements that sometimes reveal anomalies.

The models selected are the following:

#### Open-Vocabulary Object Detection Models:
- Grounded-SAM2

#### Visual Language Models:
- Qwen2-VL
- CLIP
- BLIP-2

#### Closed-Vocabulary Object Detection Models:
- YOLO11
- OneFormer

## Datasets

This repository includes two datasets designed for testing anomaly detection models in the context of autonomous trench excavation: an image dataset and a video dataset. These datasets were created from recordings of a manually operated excavator digging a trench while anomalies were intentionally revealed. The image dataset was used for initial model testing, with the most promising models further evaluated on the video dataset.

### Image Dataset

The image dataset consists of 595 selected frames. Each image is annotated in the data/annotations.json file, which has the following structure:

- Name of the image: \<name\>.png
- Anomalies in Trench: \[List of anomalies]
- Trench Boundingbox: \[xmin, xmax, ymin, ymax]


### Video Dataset

The video dataset contains 19 distinct videos, each capturing a single digging movement. These videos reveal an anomaly in the trench if there is one present. Each video is accompanied by its own annotations.json file, which follows the same structure as the one for the image dataset but provides annotations for every frame in the video:

- Name of the frame: \<name\>.png
- Anomalies in Trench: \[List of anomalies]
- Trench Boundingbox: \[xmin, xmax, ymin, ymax]

### Notes

Ensure that the folder containing the images or video frames for inference also includes the corresponding annotations.json file.

## Scripts

This section provides details on the scripts used for model testing and inference on the datasets. Both the image and video datasets are provided as images, with the video dataset being a sequence of ordered frames that can be concatenated back into a video. Each script is associated with a specific model. The required packages are installed in an Anaconda environment. Each subsection outlines the environment setup and provides examples of how to run the script. The following commands assume the image dataset is stored under `./data` and the video dataset under `./videodata`. The outputs of the respective models are saved in the `./outputs` folder.

### Grounded-SAM2:

The scripts `grounded_sam2.py` and `grounded_sam2_finetune.py` take the same input arguments. The key distinction is that `grounded_sam2_finetune.py` filters out excavator detections and objects misclassified as other objects on the excavator. The model outputs both the annotated images and a `predictions.json` file, which contains all detected objects with their corresponding bounding boxes and segmentation masks for each image.

<details>
<summary>Input Arguments</summary>

| Parameter | Default | Description |
| -| - | - |
| `groundingdino-model-config` | `configs/`<br>`groundingdino/`<br>`GroundingDINO_SwinT_OGC.py` | Specifies the configuration file for the Grounding-DINO model. |
| `groundingdino-checkpoint` | `ckpts/`<br>`grounding_dino/`<br>`groundingdino_swint_ogc.pth` | Specifies the path to the pre-trained weights for the Grounding-DINO model. |
| `sam2-model-config` | `configs/`<br>`sam2.1/`<br>`sam2.1_hiera_l.yaml` | Specifies the configuration file for the SAM2 model. |
| `sam2-checkpoint` | `ckpts/`<br>`grounded_sam2/`<br>`sam2.1_hiera_large.pt` | Specifies the path to the pre-trained weights for the SAM2 model. |
| `text-prompt` | `"pipe. `<br>`shovel. `<br>`cable. `<br>`tool. `<br>`tube. `<br>`single large stone. `<br>`barrier."` | A text prompt containing objects of interest for the model to detect. Should be a string of class names, separated by '.' and in lowercase. |
| `prompt-engineering`| `True`| When enabled uses the classes separated by '.' out of the text-prompt and puts them into context sentences. Here it is \["a {classname} laying in a trench."\]. |
| `token-spans` | `None` | Defines the start and end positions of phrases of interest within the `--text-prompt`. For example, to detect a specific word or phrase, provide its token positions. |
| `input-dir` | `data/` | The directory containing the input images. |
| `output-dir` | `outputs/`<br>`GroundedSAM2/` | Specifies the directory where the model's outputs (processed images and predictions) will be saved. |
| `box-threshold` | `0.30` | A threshold for bounding box confidence. Determines the minimum confidence required for bounding boxes to be included. |
| `text-threshold` | `0.30` | A threshold for object class name confidence. Matches tokens from the `--text-prompt` to the predicted bounding boxes with confidence > `text-threshold`. |
| `saturation` | `1.0` | Adjusts the saturation level of the output image. A value of `1.0` keeps the original saturation, while higher or lower values increase or decrease it. |
| `contrast` | `1.0` | Adjusts the contrast level of the output image. A value of `1.0` keeps the original contrast, while higher or lower values increase or decrease it. |
| `sharpness` | `1.0` | Adjusts the sharpness level of the output image. A value of `1.0` keeps the original sharpness, while higher or lower values increase or decrease it. |
| `force-cpu` | `False` | When enabled, forces the program to run on the CPU, even if a GPU is available. |

</details>


<details>
<summary>Environment Setup</summary>

To set up the Anaconda environment:

```
conda create -yn gsam2 python=3.10
conda activate gsam2
conda install -y nvidia/label/cuda-12.1.0::cuda
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install transformers supervision pycocotools

```

Download the Grounding-DINO and SAM2 pre-trained checkpoints:

```
mkdir ckpts
cd ckpts
mkdir grounding_dino
mkdir grounded_sam2
cd grounding_dino
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../grounded_sam2
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```
</details>

<details>
<summary>Inference</summary>

Run inference on the image dataset:

```
python scripts/grounded_sam2_finetune.py --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt  "pipe. shovel. cable. tool. wire. tube. single large rock. construction barrier."

```

Run inference on a single video from the video dataset:

```
python scripts/grounded_sam2_finetune.py --input-dir videodata/with_objects/01/ --box-threshold 0.25 --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt  "pipe. shovel. cable. tool. wire. tube. single large rock. construction barrier."
```
</details>

### Qwen2-VL

The script `qwen2_vl.py` provides a Python implementation for running inference with the Qwen2-VL model. The output generated by the model is a `predictions.json` file, which includes all detected objects represented as lists of their respective class names for each image.

<details>
<summary>Input Arguments</summary>

| Parameter | Default | Description |
|-|-|-|
| `qwen2-model` | `"Qwen/`<br>`Qwen2-VL-7B-Instruct"` | Specifies the path or name of the OneFormer model to be used. |
| `text-prompt` | `"Detect the anomalies `<br>`in the trench."` | A text prompt instruction telling the model what to do with the input image. |
| `input-dir` | `"data/"` | Directory containing the input images for inference. |
| `output-dir` | `"outputs/`<br>`Qwen2VL/"` | Directory where the processed images and predictions will be saved. |
| `rerun` | `False` | When set to `True`, reruns the script even if the output already exists. |
| `saturation` | `1.0` | Adjusts the saturation level of the input images. A value of `1.0` keeps the original saturation. |
| `contrast` | `1.0` | Adjusts the contrast level of the input images. A value of `1.0` keeps the original contrast. |
| `sharpness` | `1.0` | Adjusts the sharpness level of the input images. A value of `1.0` keeps the original sharpness.|
| `force-cpu` | `False` | When set to `True`, forces the program to run on the CPU even if a GPU is available. |

</details>

<details>
<summary>Environment Setup</summary>

To set up the Anaconda environment:

```
conda create -yn qwen2vl python=3.10
conda activate qwen2vl
pip install git+https://github.com/huggingface/transformers.git accelerate bitsandbytes
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
MAX_JOBS=2 python -m pip -v install --use-pep517 flash-attn --no-build-isolation
pip install qwen-vl-utils[decord] Pillow opencv-python
```

</details>

<details>
<summary>Inference</summary>

Run inference on the image dataset:

```
python scripts/qwen2_vl.py --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."
```

Run inference on a single video from the video dataset:

```
python scripts/qwen2_vl.py --input-dir videodata/with_objects/01/ --contrast 3.0 --sharpness 3.0 --saturation 2.0 \
--text-prompt "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."
```
</details>

### CLIP

The script `clip.py` contains the implementation for the CLIP model. The output is saved in a `predictions.json` file, which lists all detected objects for each image along with their class names and confidence scores.

<details>
<summary>Input Arguments</summary>

| Parameter | Default | Description |
|-|-|-|
| `clip-model` | `"openai/`<br>`clip-vit-large-patch14"` | Specifies the path or name of the OneFormer model to be used. |
| `text-prompt` | `"pipe. `<br>`shovel. `<br>`excavator shovel. `<br>`cable. `<br>`tool. `<br>`tube. `<br>`single large rock. `<br>`construction barrier. `<br>`excavator."` | A text prompt containing objects of interest for the model to detect. Should be a string of class names, separated by '.' and in lowercase. |
| `prompt-engineering`| `True`| When enabled uses the classes separated by '.' out of the text-prompt and puts them into context sentences. |
| `input-dir` | `"data/"` | Directory containing the input images for inference. |
| `output-dir` | `"outputs/`<br>`Clip/"` | Directory where the processed images and predictions will be saved. |
| `rerun` | `False` | When set to `True`, reruns the script even if the output already exists. |
| `saturation` | `1.0` | Adjusts the saturation level of the input images. A value of `1.0` keeps the original saturation. |
| `contrast` | `1.0` | Adjusts the contrast level of the input images. A value of `1.0` keeps the original contrast. |
| `sharpness` | `1.0` | Adjusts the sharpness level of the input images. A value of `1.0` keeps the original sharpness.|
| `force-cpu` | `False` | When set to `True`, forces the program to run on the CPU even if a GPU is available. |

</details>

<details>
<summary>Environment Setup</summary>

To set up the Anaconda environment:

```
conda create -yn clip python=3.10
conda activate clip
pip install git+https://github.com/huggingface/transformers.git
pip install Pillow
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

</details>

<details>
<summary>Inference</summary>

Run inference on the image dataset:

```
python scripts/clip.py --threshold 0.25 --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt  "pipe. shovel. cable. tool. wire. tube. single large rock. construction barrier."
```

</details>

### BLIP-2

The script `blip2.py` contains the implementation for the BLIP-2 model. The output is saved in a `predictions.json` file, which stores the generated image caption for each image.

<details>
<summary>Input Arguments</summary>

| Parameter | Default | Description |
|-|-|-|
| `blip2-model` | `"Salesforce/`<br>`blip2-opt-2.7b"` | Specifies the path or name of the OneFormer model to be used. |
| `text-prompt` | `"This is a picture `<br>`of a trench that has`<br>` been dug by an excavator. `<br>`The trench contains `<br>`small rocks, dirt"` | A text prompt containing  either a partially specified image caption that the model tries to complete or an empty string, indicating that the model will create a caption on its own. |
| `input-dir` | `"data/"` | Directory containing the input images for inference. |
| `output-dir` | `"outputs/`<br>`Blip2/"` | Directory where the processed images and predictions will be saved. |
| `rerun` | `False` | When set to `True`, reruns the script even if the output already exists. |
| `saturation` | `1.0` | Adjusts the saturation level of the input images. A value of `1.0` keeps the original saturation. |
| `contrast` | `1.0` | Adjusts the contrast level of the input images. A value of `1.0` keeps the original contrast. |
| `sharpness` | `1.0` | Adjusts the sharpness level of the input images. A value of `1.0` keeps the original sharpness.|
| `force-cpu` | `False` | When set to `True`, forces the program to run on the CPU even if a GPU is available. |

</details>

<details>
<summary>Environment Setup</summary>

To set up the Anaconda environment:

```
conda create -yn blip2 python=3.10
pip install git+https://github.com/huggingface/transformers.git accelerate bitsandbytes
```

</details>

<details>
<summary>Inference</summary>

Run inference on the image dataset:

```
python scripts/blip2.py --contrast 2.0 --sharpness 2.0 
```
</details>

### OneFormer

The script `oneformer.py` is the Python implementation for the OneFormer model. The outputs include both the annotated images and a `predictions.json` file, which contains all detected objects along with their corresponding segmentation masks for each image.

<details>
<summary>Input Arguments</summary>

| Parameter | Default | Description |
|-|-|-|
| `oneformer-model` | `"shi-labs/`<br>`oneformer_ade20k_dinat_large"` | Specifies the path or name of the OneFormer model to be used. |
| `input-dir` | `"data/"` | Directory containing the input images for inference. |
| `output-dir` | `"outputs/`<br>`OneFormer/"` | Directory where the processed images and predictions will be saved. |
| `rerun` | `False` | When set to `True`, reruns the script even if the output already exists. |
| `saturation` | `1.0` | Adjusts the saturation level of the input images. A value of `1.0` keeps the original saturation. |
| `contrast` | `1.0` | Adjusts the contrast level of the input images. A value of `1.0` keeps the original contrast. |
| `sharpness` | `1.0` | Adjusts the sharpness level of the input images. A value of `1.0` keeps the original sharpness.|
| `force-cpu` | `False` | When set to `True`, forces the program to run on the CPU even if a GPU is available. |

</details>

<details>
<summary>Environment Setup</summary>

To set up the Anaconda environment:

```
conda create -yn oneformer python=3.10
conda activate oneformer
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/huggingface/transformers.git
pip3 install natten==0.17.3+torch200cu118 -f https://shi-labs.com/natten/wheels
pip install scipy "numpy<2.0"
```

</details>

<details>
<summary>Inference</summary>

Run inference on the image dataset:

```
NATTEN_LOG_LEVEL=critical python scripts/oneformer.py
```

</details>

### YOLO11

The script `yolo11.py` is the Python implementation for the Ultralytics YOLO11 model. The outputs include both the annotated images and a `predictions.txt` file, which contains all detected objects along with their corresponding bounding boxes and confidence scores for each image.

<details>
<summary>Input Arguments</summary>

| Parameter | Default | Description |
|-|-|-|
| `yolo11-model` | `"yolo11x.pt"` | Specifies the path or name of the OneFormer model to be used. |
| `input-dir` | `"data/"` | Directory containing the input images for inference. |
| `output-dir` | `"outputs/`<br>`Yolo11/"` | Directory where the processed images and predictions will be saved. |
| `rerun` | `False` | When set to `True`, reruns the script even if the output already exists. |
| `saturation` | `1.0` | Adjusts the saturation level of the input images. A value of `1.0` keeps the original saturation. |
| `contrast` | `1.0` | Adjusts the contrast level of the input images. A value of `1.0` keeps the original contrast. |
| `sharpness` | `1.0` | Adjusts the sharpness level of the input images. A value of `1.0` keeps the original sharpness.|
| `force-cpu` | `False` | When set to `True`, forces the program to run on the CPU even if a GPU is available. |

</details>

<details>
<summary>Environment Setup</summary>

To set up the Anaconda environment:

```
conda create -n yolo11 python=3.10
pip install ultralytics
```

</details>

<details>
<summary>Inference</summary>

Run inference on the image dataset:

```
python scripts/yolo11.py
```

</details>