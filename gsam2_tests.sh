#!/bin/bash

python scripts/grounded_sam2.py --box-threshold 0.35 --contrast 1.0 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "

python scripts/grounded_sam2.py --box-threshold 0.35 --contrast 2.0 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "

python scripts/grounded_sam2.py --box-threshold 0.35 --contrast 0.5 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "

python scripts/grounded_sam2.py --box-threshold 0.25 --contrast 2.0 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "

python scripts/grounded_sam2.py --box-threshold 0.35 --contrast 2.0 --sharpness 2.0 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "

python scripts/grounded_sam2.py --box-threshold 0.25 --contrast 2.0 --sharpness 2.0 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "

python scripts/grounded_sam2.py --box-threshold 0.25 --contrast 2.0 --sharpness 2.0 --saturation 1.5 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "

python scripts/grounded_sam2.py --box-threshold 0.25 --contrast 2.0 --sharpness 2.0 --saturation 0.5 \
--text-prompt  "pipe. shovel. excavator shovel. cable. tool. wire. tube. a single large rock laying in the trench. construction barrier. an excavator digging a trench." \
--token-spans  " [[[0, 4]], [[6, 12]], [[32, 37]], [[39, 43]], [[45, 49]], [[51, 55]], [[59, 65], [66, 71], [72, 76]], [[14, 23], [24, 30]]] "