#!/bin/bash
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0,1,2,3  python -u train.py --name fd-animerun --stage animerun --validation animerun --restore_ckpt weights/FlowDiffuser-things.pth --gpus 0 1 2 3 --num_steps 100000 --batch_size 8 --lr 0.000001 --image_size 416 944 --wdecay 0.00001 --gamma=0.85
