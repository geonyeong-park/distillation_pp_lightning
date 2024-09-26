#python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcm_orig'
#python -m examples.run_mscoco --method "random" --model "lcmlora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcmlora_orig'
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcmlora_pp'
#python -m examples.run_mscoco --method "euler_lightning" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lightlora'
#python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lightlora_pp0.03'
#python -m examples.run_mscoco --method "euler_lightning" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/light'
#python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/light_pp'
python -m examples.run_mscoco --method "euler_a++" --model "sdxl_turbo" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/turbo' --guide_step 0 --teacher_guidance 0.
python -m examples.run_mscoco --method "euler_a++" --model "sdxl_turbo" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/turbo_pp_guide2_0.02' --guide_step 2 --teacher_guidance 0.02
python -m examples.run_mscoco --method "euler_a++" --model "sdxl_turbo" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/turbo_pp_guide2_0.1' --guide_step 2 --teacher_guidance 0.1
