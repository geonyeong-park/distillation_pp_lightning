#python -m examples.run_mscoco --method "random++_lcm" --model "lcm" --cfg_guidance 8 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcm_orig'
#python -m examples.run_mscoco --method "random" --model "lcmlora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcmlora_orig'
#python -m examples.run_mscoco --method "random++" --model "lcmlora" --cfg_guidance 8 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lcmlora_pp'
#python -m examples.run_mscoco --method "euler_lightning" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lightlora'
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/pixart/lightlora_guide0' --guide_step 0 --prompt_dir examples/assets/pixart_prompts.txt
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/pixart/lightlora_guide1' --guide_step 1 --prompt_dir examples/assets/pixart_prompts.txt
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/pixart/lightlora_guide2' --guide_step 2 --prompt_dir examples/assets/pixart_prompts.txt
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/pixart/lightlora_guide3' --guide_step 3 --prompt_dir examples/assets/pixart_prompts.txt

python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/pixart/light_guide0' --guide_step 0 --prompt_dir examples/assets/pixart_prompts.txt
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/pixart/light_guide1' --guide_step 1 --prompt_dir examples/assets/pixart_prompts.txt
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/pixart/light_guide2' --guide_step 2 --prompt_dir examples/assets/pixart_prompts.txt
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/pixart/light_guide3' --guide_step 3 --prompt_dir examples/assets/pixart_prompts.txt

python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lightlora_guide1' --guide_step 1 
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lightlora_guide2' --guide_step 2 
python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning_lora" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/lightlora_guide3' --guide_step 3 

#python -m examples.run_mscoco --method "euler_lightning" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/light'
#python -m examples.run_mscoco --method "euler_lightning++" --model "sdxl_lightning" --cfg_guidance 1 --NFE 4 --seed 777 --workdir 'examples/workdir/mscoco/light_pp'
