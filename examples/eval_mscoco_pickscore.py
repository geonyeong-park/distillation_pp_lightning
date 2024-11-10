import argparse
from pathlib import Path
from glob import glob

import os
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch


# load model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs(prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist(), scores.cpu().tolist()

def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument('--result_dir1', type=Path, default=Path('result'))
    parser.add_argument('--result_dir2', type=Path, default=Path('result'))
    parser.add_argument('--prompt_dir', type=Path, default=Path('/home/user/research/dataset/MSCOCO/coco_v2.txt'))
    args = parser.parse_args()

    print("Loading images...")
    print(f"{args.result_dir1} vs {args.result_dir2}")
    print("*" * 50)

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
    prob1, prob2, score1, score2 = 0, 0, 0, 0

    for i, (text, img1, img2) in enumerate(zip(text_list, img_list1, img_list2)):
        imgs = [Image.open(img1), Image.open(img2)]
        probs, scores = calc_probs(text, imgs)
        prob1 += probs[0]
        prob2 += probs[1]
        score1 += scores[0]
        score2 += scores[1]
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1} samples")
    print(f"Average score1: {score1 / len(text_list)}, prob1: {prob1 / len(text_list)}")
    print(f"Average score2: {score2 / len(text_list)}, prob2: {prob2 / len(text_list)}")


if __name__ == "__main__":
    main()
