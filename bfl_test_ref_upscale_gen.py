import os, base64, requests, time
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.environ.get("BFL_API_KEY")
BASE_URL = "https://api.bfl.ai/v1"

# ---------------------------------------------------------------------------
# Reference image directories — point these at your local folders
# ---------------------------------------------------------------------------
BACKGROUND_DIR = Path("./test_refs/bg")
MODEL_DIR      = Path("./test_refs/model")


# ---------------------------------------------------------------------------
# Real-ESRGAN upscaler — loaded once on first use
# ---------------------------------------------------------------------------
_upsampler = None

def get_upsampler():
    """Lazy-load the Real-ESRGAN upsampler on first use."""
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise ImportError(
            "Real-ESRGAN not installed. Run:\n"
            "  pip install basicsr realesrgan"
        )

    weights_path = Path("weights/RealESRGAN_x4plus.pth")
    if not weights_path.exists():
        print("  Downloading RealESRGAN_x4plus weights...")
        weights_path.parent.mkdir(exist_ok=True)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        r = requests.get(url, stream=True)
        with open(weights_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("  ✅ Weights downloaded.")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=4
    )

    _upsampler = RealESRGANer(
        scale=4,
        model_path=str(weights_path),
        model=model,
        tile=512,       # safe for A10G and Mac unified memory
        tile_pad=10,
        pre_pad=0,
        half=False,     # fp32 — safe on both Mac MPS and CUDA; set True on EC2 for speed
    )

    print("  ✅ Real-ESRGAN upsampler loaded.")
    return _upsampler


def upscale_image(input_path: str, output_dir: str, model_name: str, strength: float = 1.0) -> str:
    """
    Run a 4× Real-ESRGAN pass on the image at input_path.
    Saves result as {model_name}_upscale_{timestamp}.jpg and returns the path.
    
    strength: 0.0 = original, 1.0 = full upscale, 0.5 = blend
    """
    import cv2

    upsampler = get_upsampler()

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image at {input_path}")

    upscaled, _ = upsampler.enhance(img, outscale=4)

    if strength < 1.0:
        # Resize original up to match upscaled dimensions for blending
        original_resized = cv2.resize(img, (upscaled.shape[1], upscaled.shape[0]), 
                                      interpolation=cv2.INTER_LANCZOS4)
        output = cv2.addWeighted(upscaled, strength, original_resized, 1.0 - strength, 0)
    else:
        output = upscaled

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = f"{output_dir}/{model_name}_upscale_{timestamp}.jpg"
    cv2.imwrite(out_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return out_path


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def load_refs_from_dirs(background_dir: Path, model_dir: Path) -> dict:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def first_image(directory: Path) -> Path | None:
        if not directory.exists():
            print(f"  ⚠️  Directory not found: {directory}")
            return None
        matches = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if not matches:
            print(f"  ⚠️  No images found in: {directory}")
            return None
        return matches[0]

    def to_b64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    refs = {}
    bg_path    = first_image(background_dir)
    model_path = first_image(model_dir)

    if bg_path:
        print(f"  Background ref : {bg_path.name}")
        refs["input_image"] = to_b64(bg_path)

    if model_path:
        print(f"  Model ref      : {model_path.name}")
        key = "input_image_2" if "input_image" in refs else "input_image"
        refs[key] = to_b64(model_path)

    return refs


def generate(prompt, model="flux-2-pro", width=1024, height=1024, **kwargs):
    response = requests.post(
        f"{BASE_URL}/{model}",
        headers={"x-key": API_KEY, "Content-Type": "application/json"},
        json={"prompt": prompt, "width": width, "height": height, **kwargs},
    )
    response.raise_for_status()
    polling_url = response.json()["polling_url"]

    timeout_secs = 300
    start        = time.time()
    attempt      = 0

    while True:
        result  = requests.get(polling_url, headers={"x-key": API_KEY}).json()
        status  = result["status"]
        attempt += 1

        if attempt % 5 == 0 or attempt == 1:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] Status: {status}")

        if status == "Ready":
            return result["result"]["sample"]
        elif status == "Error":
            raise Exception(f"Generation failed: {result}")
        elif time.time() - start > timeout_secs:
            raise TimeoutError(f"Job timed out after {timeout_secs}s. Last status: {result}")

        time.sleep(2)


def download_image(url: str, output_dir: str = "bfl_outputs", model: str = "flux-2") -> str:
    """Download image from BFL CDN URL and save locally."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{output_dir}/{model}_{timestamp}.jpg"
    img_data  = requests.get(url).content
    with open(filename, "wb") as f:
        f.write(img_data)
    return filename


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_test(label: str, prompt: str, model: str = "flux-2-pro",
             use_refs: bool = False, use_upscale: bool = False, **kwargs):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Model: {model}  |  Refs: {use_refs}  |  Upscale: {use_upscale}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if kwargs:
        print(f"Params: { {k: v for k, v in kwargs.items() if not k.startswith('input_image')} }")
    print(f"{'='*60}")

    output_dir = "bfl_outputs"
    start      = time.time()

    try:
        extra = {}
        if use_refs:
            extra = load_refs_from_dirs(BACKGROUND_DIR, MODEL_DIR)
            if not extra:
                print("  ⚠️  No reference images loaded — running text-only")

        image_url  = generate(prompt, model=model, **kwargs, **extra)
        local_path = download_image(url=image_url, output_dir=output_dir, model=model)
        elapsed    = time.time() - start

        print(f"✅ Generation done in {elapsed:.1f}s")
        print(f"   CDN URL   : {image_url}")
        print(f"   Saved to  : {local_path}")

        result = {
            "label":        label,
            "status":       "ok",
            "path":         local_path,
            "upscaled_path": None,
            "seconds":      elapsed,
        }

        if use_upscale:
            print(f"\n  Running Real-ESRGAN 4× upscale...")
            upscale_start = time.time()
            upscaled_path = upscale_image(
                input_path=local_path,
                output_dir=output_dir,
                model_name=model,
                strength=kwargs.pop("upscale_strength", 1.0),
            )
            upscale_elapsed = time.time() - upscale_start
            print(f"  ✅ Upscale done in {upscale_elapsed:.1f}s")
            print(f"     Saved to : {upscaled_path}")
            result["upscaled_path"] = upscaled_path

        return result

    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Failed after {elapsed:.1f}s: {e}")
        return {"label": label, "status": "error", "error": str(e), "seconds": elapsed}


# ---------------------------------------------------------------------------
# Test suite — edit freely
# ---------------------------------------------------------------------------
TEST_CASES = [
    # {
    #     "label": "Pro — basic portrait",
    #     "model": "flux-2-pro",
    #     "prompt": "Realistic studio portrait of a woman, soft box lighting, clean white background, "
    #               "sharp focus, commercial photography",
    #     "width": 1920,
    #     "height": 1080,
    # },
    # {
    #     "label": "Pro — product shot",
    #     "model": "flux-2-pro",
    #     "prompt": "Luxury perfume bottle on a white marble surface, specular highlights, "
    #               "bokeh background, editorial product photography",
    #     "width": 1280,
    #     "height": 1280,
    # },
    # {
    #     "label": "Max — high-fidelity portrait",
    #     "model": "flux-2-max",
    #     "prompt": "Close-up beauty shot of a woman, glass-skin dewy glow, natural makeup, "
    #               "shallow depth of field, luxury fashion editorial",
    #     "width": 1920,
    #     "height": 1080,
    # },
    {
        "label": "Max — multi-reference (bg + model) 1",
        "model": "flux-2-max",
        "use_refs": True,
        "use_upscale": True,
        "upscale_strength": 0.8,
        "prompt": """
        Professional urban lifestyle marketing photograph in natural outdoor lighting with sharp focus, high resolution, 8k detail, photorealistic. 
        Young Black male model with short natural hair and athletic build similar to image 2 stands leaning casually with back against the concrete bridge railing in the location from image 1, 
        wearing olive green and tan camouflage hoodie with matching camouflage joggers and black athletic sneakers similar to image 2. Relaxed, confident expression. Urban street style commercial photography.
        """,
        "width": 1920,
        "height": 1080,
    },
]

if __name__ == "__main__":
    if not API_KEY:
        raise EnvironmentError("BFL_API_KEY not set — check your .env file.")

    print(f"Running {len(TEST_CASES)} test(s) against BFL API...")
    print(f"Reference dirs:")
    print(f"  Backgrounds : {BACKGROUND_DIR.resolve()}")
    print(f"  Models      : {MODEL_DIR.resolve()}")

    results = [run_test(**tc) for tc in TEST_CASES]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        icon = "✅" if r["status"] == "ok" else "❌"
        if r["status"] == "ok":
            print(f"{icon} [{r['seconds']:.1f}s] {r['label']}")
            print(f"     Original : {r['path']}")
            if r.get("upscaled_path"):
                print(f"     Upscaled : {r['upscaled_path']}")
        else:
            print(f"{icon} [{r['seconds']:.1f}s] {r['label']}: {r.get('error', '')}")