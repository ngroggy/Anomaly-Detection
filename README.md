# Anomaly Detection

An Excavator is digging a trench I have to detect if some objects other than dirt or small stones are present in the trench. If there are any objects, e.g. pipes, tools, large stones or other construction site objects, present in the trench, the model should detect it. The models selected are:

yolov11
Grounded-SAM2
(DITO)
OneFormer
BLIP2
Qwen2VL
(Clip)

#### Dataset

Approximately 600 images scraped from a videos of an excavator digging a trench in Bremgarten. The digging process was sometimes paused to put in objects/anoamlies inside the trench that would prevent the excavator from digging further.

Every image has an entry in the annotations.json which is structured as following:

Name of the image: \<name\>.png
Anomalies Present: True/False [1/0]
Anomalies in Trench: \[List of anomalies]
Trench Boundingbox: \[xmin, xmax, ymin, ymax]

Make sure data/ folder contains the annotations.json file for inference.

#### Output of the models:

Is there an anomaly present: True/False
Which anomalies are present: \[List of anomalies]
Where are the anomalies: \[List of bboxes]

#### Evaluations:

Is there an anomaly present: 
- \# and \% of correct answers 
- \# and \% of true positives and negatives
- \# and \% of false positives
- \# and \% of false negatives

Which anomalies are present:
- Are the object classes correctly identified
- Are correct number of objects identified

Where are anomalies located:
- Are the positions of the detected objects approximately correct

#### Testing Parameters for optimization

Various factors are tested to get the best outcome of the models:
- Saturation, Contrast, Sharpness
- Image Cropping
- Blacking out background

## Model Usage

### Grounded-SAM2:

```
python utils/groundedsam2/tools/generate_tokenspans.py --text-prompt "pipe. shovel. excavator shovel. cable. tool. wire. tube. construction barrier. an excavator digging a trench with a large rock inside."  --classes "pipe, shovel, cable, tool, wire, tube, large rock, excavator shovel"
```


```
python scripts/grounded_sam2.py --box-threshold 0.35 --contrast 2.0 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "

```

### OneFormer

```
NATTEN_LOG_LEVEL=critical python scripts/oneformer.py
```

## Miscellaneous 

### Prompts used:

#### Grounded-SAM2
```
--text-prompt "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large boulder laying in the trench. construction barrier. an excavator digging a trench."  --classes "pipe, shovel, cable, tool, wire, tube, large boulder, barrier"
```
```
--text-prompt "an excavator is digging a trench. the different objects lying in the trench are a pipe, cable, shovel, tool, single large stone, barrier or a tube." --token-span "[[[82, 86]], [[88, 93]], [[95, 101]], [[103, 107]], [[116, 121], [122, 127]], [[129, 136]], [[142, 146]]]"
```
```
"an excavator is digging a trench. a pipe is lying on dirt. a pipe is covered in dirt. a pipe is sticking out of dirt. a shovel is lying on dirt. a shovel is covered in dirt. a shovel is sticking out of dirt. a tool is lying on dirt. a tool is covered in dirt. a tool is sticking out of dirt. a single large stone is lying on dirt. a single large stone is covered in dirt. a large stone is sticking out of dirt. a construction barrier is lying on dirt. a construction barrier is covered in dirt. a construction barrier is sticking out of dirt. a tube is lying on dirt. a tube is covered in dirt. a tube is sticking out of dirt. a cable is lying on dirt. a cable is covered in dirt. a cable is sticking out of dirt. "
```
```
"an excavator is digging a trench. there is an object other than small stones or dirt inside the trench."
```

#### Qwen2VL
```
"This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, little rocks, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."
```
```
"This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."
```
```
"This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."
```

### Environment Setup

#### Grounded-SAM2
```
conda create -yn gsam2 python=3.10
conda activate gsam2
conda install -y nvidia/label/cuda-12.1.0::cuda
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install transformers supervision pycocotools

```
#### Yolo11

```
conda create -n yolo11 python=3.10
pip install ultralytics
```

#### OneFormer

```
conda create -yn oneformer python=3.10
conda activate hf_oneformer
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/huggingface/transformers.git
pip3 install natten==0.17.3+torch200cu118 -f https://shi-labs.com/natten/wheels
pip install scipy "numpy<2.0"
```


#### blip2

```
conda create -yn blip2 python=3.10
conda activate blip2
pip install git+https://github.com/huggingface/transformers.git accelerate bitsandbytes
pip install pillow
```