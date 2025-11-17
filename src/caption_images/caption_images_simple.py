# caption_images.py
from pathlib import Path

dataset_dir = Path("./data/small_set_style1")

# Simple template caption with trigger word
# template = "[brandname]style, professional photography, high quality"
# template = "DSG_style_1, model_1, professional photography, high quality"
template = "DSG_style_A, DSG_model_1, DSG_sweatshirt_blue, DSG_pants_dark, walking, checkered slip-on shoes"

# Create .txt file for each image
for img_path in dataset_dir.glob("*.jpg"):
    caption_path = img_path.with_suffix(".txt")
    caption_path.write_text(template)
    print(f"Created: {caption_path.name}")

print("Captioning complete!")