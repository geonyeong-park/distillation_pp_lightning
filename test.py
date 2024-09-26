from diffusers import AutoPipelineForText2Image
import torch
import numpy as np

torch.random.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)

pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline_text2image = pipeline_text2image.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

image = pipeline_text2image(prompt=prompt, guidance_scale=0.0, num_inference_steps=4).images[0]
image.save("examples/workdir/t2i/result/generated_test.png")

"""
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler, StableDiffusionXLPipeline
import torch
import numpy as np

torch.random.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "an astronaut sitting in a diner, eating fries, cinematic, analog film"

generator = torch.manual_seed(777)
image = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
image.save("examples/workdir/t2i/result/generated_test.png")
"""