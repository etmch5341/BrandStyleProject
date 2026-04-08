import os, requests, time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("BFL_API_KEY")
BASE_URL = "https://api.bfl.ai/v1"

def generate(prompt, model="flux-2-pro", width=1024, height=1024, **kwargs):
    """
    model options:
      "flux-2-klein"  — drafts/previews (fast, cheap)
      "flux-2-pro"    — production standard
      "flux-2-max"    — premium output (print/luxury)
    """
    # Submit job
    response = requests.post(
        f"{BASE_URL}/{model}",
        headers={"x-key": API_KEY, "Content-Type": "application/json"},
        json={"prompt": prompt, "width": width, "height": height, **kwargs}
    )
    response.raise_for_status()
    polling_url = response.json()["polling_url"]

    # Poll until ready
    while True:
        result = requests.get(polling_url, headers={"x-key": API_KEY}).json()
        if result["status"] == "Ready":
            return result["result"]["sample"]  # image URL — download immediately
        elif result["status"] == "Error":
            raise Exception(f"Generation failed: {result}")
        time.sleep(1)


def download_image(url: str, output_dir: str = "bfl_outputs") -> str:
    """Download image from BFL CDN URL and save locally. URLs expire quickly — download immediately."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{timestamp}.jpg"
    img_data = requests.get(url).content
    with open(filename, "wb") as f:
        f.write(img_data)
    return filename


def run_test(label: str, prompt: str, model: str = "flux-2-pro", **kwargs):
    """Run a single generation test, download result, and print a summary."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"Params: {kwargs}")
    print(f"{'='*60}")

    start = time.time()
    try:
        image_url = generate(prompt, model=model, **kwargs)
        elapsed = time.time() - start
        local_path = download_image(image_url)
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
    {
        "label": "Pro — basic portrait",
        "model": "flux-2-pro",
        "prompt": "Realistic studio portrait of a woman, soft box lighting, clean white background, "
                  "sharp focus, commercial photography",
        "width": 1920,
        "height": 1080,
    },
    {
        "label": "Pro — product shot",
        "model": "flux-2-pro",
        "prompt": "Luxury perfume bottle on a white marble surface, specular highlights, "
                  "bokeh background, editorial product photography",
        "width": 1280,
        "height": 1280,
    },
    {
        "label": "Max — high-fidelity portrait",
        "model": "flux-2-max",
        "prompt": "Close-up beauty shot of a woman, glass-skin dewy glow, natural makeup, "
                  "shallow depth of field, luxury fashion editorial",
        "width": 1920,
        "height": 1080,
    },
]

if __name__ == "__main__":
    if not API_KEY:
        raise EnvironmentError("BFL_API_KEY environment variable is not set.")

    print(f"Running {len(TEST_CASES)} test(s) against BFL API...")
    results = [run_test(**tc) for tc in TEST_CASES]

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status_icon = "✅" if r["status"] == "ok" else "❌"
        detail = r.get("path", r.get("error", ""))
        print(f"{status_icon} [{r['seconds']:.1f}s] {r['label']}: {detail}")