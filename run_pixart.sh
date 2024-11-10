#python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcm_orig'
#python -m examples.run_mscoco --method "random" --model "lcmlora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcmlora_orig'
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcmlora_pp'
#python -m examples.run_mscoco --method "euler_lightning" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lightlora'
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/pixart/lcmlora_guide0' --guide_step 0 --prompt_dir examples/assets/pixart_prompts.txt
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 3 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/pixart/lcmlora_guide1' --guide_step 1 --prompt_dir examples/assets/pixart_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_orig' --guide_step 0 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_guide1_0.01' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_guide2_0.01' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_guide3_0.01' --guide_step 3 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_guide4_0.01' --guide_step 4 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_guide1_0.02' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_guide2_0.02' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_guide3_0.02' --guide_step 3 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcm_guide4_0.02' --guide_step 4 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 6 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcmlora_orig' --guide_step 0 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0
python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 6 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcmlora_guide1_0.02' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 6 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcmlora_guide2_0.02' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 6 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcmlora_guide3_0.02' --guide_step 3 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 6 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lcmlora_guide4_0.02' --guide_step 4 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_orig' --guide_step 0 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide1_0.1' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.1
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide1_0.15' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.15
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide1_0.05' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.05
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide1_0.02' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide1_0.01' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide2_0.1' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.1
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide2_0.15' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.15
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide2_0.05' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.05
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide2_0.02' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/light_guide2_0.01' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_orig' --guide_step 0 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_guide1_0.01' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_guide1_0.02' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_guide1_0.05' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.05
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_guide1_0.1' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.1
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_guide2_0.1' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.1
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_guide2_0.01' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.01
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_guide2_0.02' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.02
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/lightlora_guide2_0.05' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.05
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_orig' --guide_step 0 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide1_0.05' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.05
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide1_0.07' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.07
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide1_0.1' --guide_step 1 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.1
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide2_0.1' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.1
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide2_0.05' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.05
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide2_0.07' --guide_step 2 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.07
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide3_0.07' --guide_step 3 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.07
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide3_0.05' --guide_step 3 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.05
python -m examples.run_mscoco --method "dpm++_2s_a_cfgpp++" --model "sdxl_turbo" --cfg_guidance 1.2 --NFE 6 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/curated_prompts/turbo_guide3_0.1' --guide_step 3 --prompt_dir examples/assets/curated_prompts.txt --teacher_guidance 0.1




#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide1' --guide_step 1 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide2' --guide_step 2 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide3' --guide_step 3 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide4' --guide_step 4 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide5' --guide_step 5 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide6' --guide_step 6 
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 8 --seed 777 --workdir '/home/user/hdisk/distillation_pp_results/mscoco/lcmlora_guide7' --guide_step 7 

#python -m examples.run_mscoco --method "euler_lightning" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/light'
#python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/light_pp'
