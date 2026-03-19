import os
import glob
import numpy as np
from PIL import Image
import onnxruntime as ort

# --- CONFIGURATION ---
INPUT_DIR = "path/to/your/brand_assets"  # Your unlabeled images
TRIGGER_WORD = "brandtype1"               # Your unique brand identifier
THRESHOLD = 0.35                          # Confidence level (0.0 to 1.0)
MODEL_PATH = "wd-v1-4-convnext-tagger-v2.onnx" # Download this from HuggingFace

def tag_images():
    # Load the ONNX model (Ensure you've downloaded the .onnx and .csv tags file)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    
    # Load tag list (usually selected_tags.csv)
    with open("selected_tags.csv", "r") as f:
        tags = [line.split(",")[1] for line in f.readlines()[1:]]

    image_files = glob.glob(os.path.join(INPUT_DIR, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(INPUT_DIR, "*.[pP][nN][gG]"))

    for img_path in image_files:
        # Preprocess Image
        img = Image.open(img_path).convert("RGB").resize((448, 448))
        img_np = np.array(img).astype(np.float32)[:, :, ::-1] # RGB to BGR
        img_np = np.expand_dims(img_np, axis=0)

        # Run Inference
        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[0].name
        probs = session.run([label_name], {input_name: img_np})[0][0]

        # Extract Tags
        found_tags = [tags[i] for i, p in enumerate(probs) if p > THRESHOLD]
        
        # Clean up tags (remove underscores, typical for WD14)
        clean_tags = [t.replace("_", " ") for t in found_tags]
        
        # Prepend Trigger Word
        final_caption = f"{TRIGGER_WORD}, " + ", ".join(clean_tags)

        # Save to .txt file
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        with open(txt_path, "w") as f:
            f.write(final_caption)
            
        print(f"Processed: {os.path.basename(img_path)}")

if __name__ == "__main__":
    tag_images()