import argparse
from pathlib import Path
from glob import glob

import os
import numpy as np
import torch
import ImageReward as RM

def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument('--result_dir1', type=Path, default=Path('result'))
    parser.add_argument('--result_dir2', type=Path, default=Path('result'))
    parser.add_argument('--prompt_dir', type=Path, default=Path('/home/user/research/dataset/MSCOCO/coco_v2.txt'))
    args = parser.parse_args()

    # load prompt
    text_list = []
    with open(args.prompt_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                text_list.append(stripped_line)
    text_list = text_list[:10000]

    img_list1 = sorted(glob(str(args.result_dir1 / '*.png')))
    img_list2 = sorted(glob(str(args.result_dir2 / '*.png')))
    reward1, reward2 = 0, 0

    model = RM.load("ImageReward-v1.0")

    for i, (text, img1, img2) in enumerate(zip(text_list, img_list1, img_list2)):
        reward1 += model.score(text, img1)
        reward2 += model.score(text, img2) 
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1} samples")
    print(f"Average reward1: {reward1 / len(text_list)}")
    print(f"Average reward2: {reward2 / len(text_list)}")


if __name__ == "__main__":
    main()