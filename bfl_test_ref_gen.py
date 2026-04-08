import os, base64, requests, time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.environ.get("BFL_API_KEY")
BASE_URL = "https://api.bfl.ai/v1"

# ---------------------------------------------------------------------------
# Reference image directories — point these at your local folders
# ---------------------------------------------------------------------------
BACKGROUND_DIR = Path("./test_refs/bg")   # edit as needed
MODEL_DIR      = Path("./test_refs/model")         # edit as needed


def load_refs_from_dirs(background_dir: Path, model_dir: Path) -> dict:
    """
    Loads the first image from each directory and returns a dict of
    input_image / input_image_2 / ... keys ready to merge into a payload.

    Ordering matches the pipeline convention:
      input_image   → background (image 1 in prompt)
      input_image_2 → model reference (image 2 in prompt)
    """
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

    timeout_secs = 300  # 5 min max
    start = time.time()
    attempt = 0

    while True:
        result = requests.get(polling_url, headers={"x-key": API_KEY}).json()
        status = result["status"]
        attempt += 1

        # Print status every 5 polls so you can see it's alive
        if attempt % 5 == 0 or attempt == 1:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] Status: {status}")

        if status == "Ready":
            return result["result"]["sample"]
        elif status == "Error":
            raise Exception(f"Generation failed: {result}")
        elif time.time() - start > timeout_secs:
            raise TimeoutError(f"Job timed out after {timeout_secs}s. Last status: {result}")

        time.sleep(2)  # slightly more generous than 1s


def download_image(url: str, output_dir: str = "bfl_outputs", model: str = "flux-2") -> str:
    """Download image from BFL CDN URL and save locally. URLs expire quickly — download immediately."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{output_dir}/{model}_{timestamp}.jpg"
    img_data  = requests.get(url).content
    with open(filename, "wb") as f:
        f.write(img_data)
    return filename


def run_test(label: str, prompt: str, model: str = "flux-2-pro",
             use_refs: bool = False, **kwargs):
    """Run a single generation test, download result, and print a summary."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if kwargs:
        print(f"Params: { {k: v for k, v in kwargs.items() if not k.startswith('input_image')} }")
    print(f"{'='*60}")

    start = time.time()
    try:
        extra = {}
        if use_refs:
            extra = load_refs_from_dirs(BACKGROUND_DIR, MODEL_DIR)
            if not extra:
                print("  ⚠️  No reference images loaded — running text-only")

        image_url  = generate(prompt, model=model, **kwargs, **extra)
        elapsed    = time.time() - start
        local_path = download_image(url=image_url, model=model)
        print(f"✅ Done in {elapsed:.1f}s")
        print(f"   CDN URL   : {image_url}")
        print(f"   Saved to  : {local_path}")
        return {"label": label, "status": "ok", "path": local_path, "seconds": elapsed}

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
    # ── Reference test ──────────────────────────────────────────────────────
    # Loads background from BACKGROUND_DIR and model from MODEL_DIR.
    # Prompt references them explicitly as "image 1" and "image 2".
    {
        "label": "Max — multi-reference (bg + model) 1",
        "model": "flux-2-max",
        "use_refs": True,
        "prompt": """
        Professional urban lifestyle marketing photograph in natural outdoor lighting with sharp focus, high resolution, 8k detail, photorealistic. 
        Young Black male model with short natural hair and athletic build similar to image 2 stands leaning casually with back against the concrete bridge railing in the location from image 1, 
        wearing olive green and tan camouflage hoodie with matching camouflage joggers and black athletic sneakers similar to image 2. Relaxed, confident expression. Urban street style commercial photography.
        """,
        "width": 1920,
        "height": 1080,
    },
    # {
    #     "label": "Pro — multi-reference (bg + model)",
    #     "model": "flux-2-max",
    #     "use_refs": True,
    #     "prompt": """
    #     Professional skincare product photography. The oil serum hybrid bottle (small rounded glass bottle, iridescent pink-to-peach gradient, soft blush-pink cap, black logo and text, 30ml 1fl oz) from image 2 red in frame, 
    #     resting against or floating above a warm dreamy background of soft glowing skin texture with iridescent peach-gold shimmer, cascading light streaks and bokeh, warm blush and cream tones from image 1. The bottle's iridescent 
    #     glass catches and reflects the warm ambient glow of the background. Extreme shallow depth of field, soft dreamy focus on background, sharp focus on product label. 
    #     Luxury beauty editorial photography, skincare editorial.
    #     """,
    #     "width": 1080,
    #     "height": 1920,
    # },
    # {
    #     "label": "Pro — multi-reference (bg + model)",
    #     "model": "flux-2-pro",
    #     "use_refs": True,
    #     "prompt": """
    #     Professional skincare product photography. The oil serum hybrid bottle (small rounded glass bottle, iridescent pink-to-peach gradient, soft blush-pink cap, black logo and text, 30ml 1fl oz) from image 2 red in frame, 
    #     resting against or floating above a warm dreamy background of soft glowing skin texture with iridescent peach-gold shimmer, cascading light streaks and bokeh, warm blush and cream tones from image 1. The bottle's iridescent 
    #     glass catches and reflects the warm ambient glow of the background. Sharp focus on product label. Make sure the bottle is clear and sharp. 
    #     Luxury beauty editorial photography, skincare editorial.
    #     """,
    #     "width": 1080,
    #     "height": 1920,
    # },
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
        icon   = "✅" if r["status"] == "ok" else "❌"
        detail = r.get("path", r.get("error", ""))
        print(f"{icon} [{r['seconds']:.1f}s] {r['label']}: {detail}")