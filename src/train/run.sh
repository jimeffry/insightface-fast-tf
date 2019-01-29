#! /bin/bash
python train.py --gpu-list 0 --save-weight-period 20  --batch-size 16 --margin-a 0.9 --margin-b 0.15 --margin-m 0.4 --scale-s 10