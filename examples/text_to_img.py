import argparse
from pathlib import Path

from munch import munchify
from torchvision.utils import save_image
import torch

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed
import time


def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/t2i")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="low quality,jpeg artifacts,blurry,poorly drawn,ugly,worst quality,")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--teacher_guidance", type=float, default=0.02)
    parser.add_argument("--guide_step", type=int, default=2)
    parser.add_argument("--method", type=str, default='ddim')
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl", "sdxl_lightning", "sdxl_lightning_lora", "lcm", "lcmlora", "dmd", "sdxl_turbo"])
    parser.add_argument("--renoise", type=str, default='random', choices=["random", "deterministic", "hybrid"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    create_workdir(args.workdir)

    solver_config = munchify({'num_sampling': args.NFE,
                              'do_lora': True if 'lora' in args.model else False,
                              'model': args.model,
                              })
    guide_config = munchify({'teacher_guidance': args.teacher_guidance,
                             'guide_step': args.guide_step,
                             'renoise': args.renoise,
                             })
    """
    callback = ComposeCallback(workdir=args.workdir,
                               frequency=1,
                               callbacks=["draw_noisy", 'draw_tweedie'],
                               seed=args.seed)
    """
    callback = None


    if args.model == "sdxl" or args.model == "sdxl_lightning" or args.model == 'sdxl_lightning_lora' or args.model == 'lcm' or args.model == 'lcmlora' or args.model == 'sdxl_turbo' or args.model == 'dmd':
        solver = get_solver_sdxl(args.method,
                                 solver_config=solver_config,
                                 device=args.device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = solver.sample(prompt1=[args.null_prompt, args.prompt],
                                prompt2=[args.null_prompt, args.prompt],
                                cfg_guidance=args.cfg_guidance,
                                target_size=(1024, 1024),
                                callback_fn=callback,
                                **guide_config)
        end_event.record()
        torch.cuda.synchronize()
        print("Time: ", start_event.elapsed_time(end_event), "ms")

    else:
        solver = get_solver(args.method,
                            solver_config=solver_config,
                            device=args.device)
        result = solver.sample(prompt=[args.null_prompt, args.prompt],
                               cfg_guidance=args.cfg_guidance,
                               callback_fn=callback)


    save_image(result, args.workdir.joinpath(f'result/generated.png'), normalize=True)

if __name__ == "__main__":
    main()
