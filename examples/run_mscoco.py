import argparse
from pathlib import Path

import os
import numpy as np
import torch
from PIL import Image

from munch import munchify
from torchvision.utils import save_image

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed


def create_workdir(workdir: Path):
    workdir.mkdir(parents=True, exist_ok=True)
    #workdir.joinpath('images').mkdir(exist_ok=True)
    #workdir.joinpath('logs').mkdir(exist_ok=True)

def set_seed(seed: int):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/t2i")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="low quality,jpeg artifacts,blurry,poorly drawn,ugly,worst quality,")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--teacher_guidance", type=float, default=0.02)
    parser.add_argument("--method", type=str, default='ddim')
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl", "sdxl_lightning", "sdxl_lightning_lora", "lcm", "lcmlora"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--result_dir', type=Path, default=Path('result'))
    parser.add_argument('--prompt_dir', type=Path, default=Path('/home/user/research/dataset/MSCOCO/coco_v2.txt'))
    args = parser.parse_args()

    set_seed(args.seed)
    create_workdir(args.workdir)

    solver_config = munchify({'num_sampling': args.NFE, 
                              'do_lora': True if 'lora' in args.model else False,
                              })
    guide_config = munchify({'teacher_guidance': args.teacher_guidance})
    callback = None

    # load prompt
    text_list = []
    with open(args.prompt_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                text_list.append(stripped_line)
    text_list = text_list[:10000]

    # load model
    solver = get_solver_sdxl(args.method,
                             solver_config=solver_config,
                             device=args.device)

    # inference
    for i, text in enumerate(text_list):
        print(f'Processing {i+1}/{len(text_list)}: {text}')

        result = solver.sample(prompt1=[args.null_prompt, text],
                                prompt2=[args.null_prompt, text],
                                cfg_guidance=args.cfg_guidance,
                                target_size=(1024, 1024),
                                callback_fn=callback,
                                **guide_config)
        fname = os.path.join(args.workdir, f'{str(i).zfill(5)}.png')
        save_image(result, fname, normalize=True)

if __name__ == '__main__':
    main()