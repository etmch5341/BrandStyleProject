from diffusers import AutoPipelineForText2Image
import torch

# 1. Load the base SDXL model
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("mps") # Use "mps" for Mac or "cuda" for NVIDIA

# 2. Load your newly trained LoRA
# Replace 'path/to/lora' with your 'lora_final' folder or the .safetensors file
pipe.load_lora_weights("./lora_output/lora_final", weight_name="pytorch_lora_weights.safetensors")

# 3. Generate an image
# Use the 'trigger words' you used in your training captions!
prompt = "A high-end marketing photo in DSG_style_A in natural lighting where DSG_model_1 is sitting down on a bench in a park"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

image.save("test_brand_output.png")

#DSG_style_A, DSG_model_1, DSG_sweatshirt_blue, DSG_pants_dark, walking, checkered slip-on shoes