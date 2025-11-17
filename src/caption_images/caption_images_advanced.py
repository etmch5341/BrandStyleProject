# auto_caption_blip.py
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Load BLIP-2 model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
).to("cuda")

dataset_dir = Path("datasets/brand_references")

for img_path in dataset_dir.glob("*.jpg"):
    # Load image
    image = Image.open(img_path).convert("RGB")
    
    # Generate caption
    inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Add trigger word
    final_caption = f"{caption}, DSG style 1, professional photography, high quality"
    
    # Save caption
    caption_path = img_path.with_suffix(".txt")
    caption_path.write_text(final_caption)
    
    print(f"{img_path.name}: {final_caption}")

print("Auto-captioning complete!")