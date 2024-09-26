#python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcm_orig'
#python -m examples.run_mscoco --method "random" --model "lcmlora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcmlora_orig'
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcmlora_pp'
#python -m examples.run_mscoco --method "euler_lightning" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lightlora'
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/pixart/lcmlora_guide0' --guide_step 0 --prompt_dir examples/assets/pixart_prompts.txt
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 3 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/pixart/lcmlora_guide1' --guide_step 1 --prompt_dir examples/assets/pixart_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcm_guide1' --guide_step 1 --teacher_guidance 0.01
python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 3 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide1' --guide_step 1 --teacher_guidance 0.01
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/light_guide1' --guide_step 1 --teacher_guidance 0.1
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lightlora_guide1' --guide_step 1 --teacher_guidance 0.1


#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide1' --guide_step 1 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide2' --guide_step 2 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide3' --guide_step 3 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide4' --guide_step 4 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide5' --guide_step 5 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide6' --guide_step 6 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide7' --guide_step 7 

#python -m examples.run_mscoco --method "euler_lightning" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/light'
#python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/light_pp'
