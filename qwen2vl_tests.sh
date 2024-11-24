#!/bin/bash

python scripts/qwen2_vl.py \
--text-prompt "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, little rocks, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."

python scripts/qwen2_vl.py \
--text-prompt "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."

python scripts/qwen2_vl.py \
--text-prompt "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."

python scripts/qwen2_vl.py --contrast 2.0 \
--text-prompt "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, little rocks, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."

python scripts/qwen2_vl.py --contrast 2.0 \
--text-prompt "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."

python scripts/qwen2_vl.py --contrast 2.0 \
--text-prompt "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'."