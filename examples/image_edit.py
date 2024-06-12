import argparse
from pathlib import Path

import numpy as np
import torch
from munch import munchify
from PIL import Image
from torchvision.utils import save_image

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed


def load_img(img_path: str, size: int=512, centered:bool=True):
    image = np.array(Image.open(img_path).convert('RGB').resize((size, size)))
    image = torch.from_numpy(image).permute(2, 0, 1)
    if centered: 
        image = image / 127.5 - 1  # [0, 1] -> [-1, 1]
    image = image.unsqueeze(0)
    return image

def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion Editing")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/edit")
    parser.add_argument("--img_path", type=Path, default="examples/assets/afhq_1.jpg")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--src_prompt", type=str, default="")
    parser.add_argument("--tgt_prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--method", type=str, default='ddim_edit')
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    create_workdir(args.workdir)

    solver_config = munchify({'num_sampling': args.NFE })
    callback = ComposeCallback(workdir=args.workdir,
                               callbacks=["draw_tweedie", "draw_noisy"])

    img = load_img(args.img_path, size=args.img_size)

    if args.model == "sdxl":
        solver = get_solver_sdxl(args.method,
                                 solver_config=solver_config,
                                 device=args.device)
        result = solver.sample(prompt1=[args.null_prompt, args.src_prompt, args.tgt_prompt],
                               prompt2=[args.null_prompt, args.src_prompt, args.tgt_prompt],
                               src_img=img,
                               cfg_guidance=args.cfg_guidance,
                               target_size=(1024, 1024),
                               callback_fn=callback)
    else:
        solver = get_solver(args.method,
                            solver_config=solver_config,
                            device=args.device)
        result = solver.sample(prompt=[args.null_prompt, args.src_prompt, args.tgt_prompt],
                               src_img=img,
                               cfg_guidance=args.cfg_guidance,
                               callback_fn=callback)
        
    save_image(result, args.workdir.joinpath(f'result/edited.png'), normalize=True)

if __name__ == "__main__":
    main()
