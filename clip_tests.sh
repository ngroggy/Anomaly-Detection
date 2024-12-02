#!/bin/bash

python scripts/clip.py --contrast 1.0 --sharpness 1.0 --saturation 1.0 

python scripts/clip.py --contrast 2.0 --sharpness 1.0 --saturation 1.0 

python scripts/clip.py --contrast 1.0 --sharpness 2.0 --saturation 1.0 

python scripts/clip.py --contrast 2.0 --sharpness 2.0 --saturation 1.5 

python scripts/clip.py --contrast 2.0 --sharpness 2.0 --saturation 0.5

python scripts/clip.py --contrast 1.0 --sharpness 1.0 --saturation 2.0 

python scripts/clip.py --threshold 0.25 --contrast 2.0 --sharpness 2.0 --saturation 1.5 

python scripts/clip.py --threshold 0.28 --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt "pipe. hose. duct. cable. tool. wire. tube. single large rock. construction barrier. excavator shovel. excavator."

python scripts/clip.py --threshold 0.28 --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt "pipe. hose. duct. cable. tool. wire. tube. single large rock. construction barrier. excavator shovel. excavator." # only clip text engineering

python scripts/clip.py --threshold 0.28 --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt "pipe. hose. duct. cable. tool. wire. tube. single large rock. construction barrier. excavator shovel. excavator." # only own text engineering

python scripts/clip.py --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt "pipe. hose. duct. cable. tool. wire. tube. single large rock. construction barrier. excavator shovel. excavator." # only own text engineering