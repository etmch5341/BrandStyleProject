"""
Brand Style Marketing Generator — FastAPI Backend
Supports: FLUX.2 Klein (local), FLUX.2-Pro (BFL API), FLUX.2-Max (BFL API)
Post-processing: Real-ESRGAN upscaling (optional)
Prompt optimization: Ollama (local LLM)
"""

import os
import gc
import re
import uuid
import time
import base64
import tempfile
import threading
import random
import logging
import httpx
import json
from io import BytesIO
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import boto3
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("brand-gen")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_KEY             = os.getenv("API_KEY", "changeme-secret-key")
S3_BUCKET           = os.getenv("S3_BUCKET", "your-brand-gen-outputs")
AWS_REGION          = os.getenv("AWS_REGION", "us-east-2")

# BFL
BFL_API_KEY         = os.getenv("BFL_API_KEY", "")
BFL_BASE_URL        = "https://api.bfl.ai/v1"
BFL_POLL_TIMEOUT    = int(os.getenv("BFL_POLL_TIMEOUT", "300"))
BFL_POLL_INTERVAL   = float(os.getenv("BFL_POLL_INTERVAL", "2.0"))

# Klein — only loaded if KLEIN_ENABLED=true
KLEIN_ENABLED       = os.getenv("KLEIN_ENABLED", "false").lower() == "true"
HF_TOKEN            = os.getenv("HF_TOKEN", "")
FLUX_REPO_ID        = "black-forest-labs/FLUX.2-klein-4B"
TORCH_DTYPE_STR     = os.getenv("TORCH_DTYPE", "bfloat16")

# Ollama
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "llama3")

CLOUDFRONT_OUTPUTS  = "https://d52unpfog765b.cloudfront.net"

# ─────────────────────────────────────────────
# Prompt Engineer System Prompt (for Ollama)
# ─────────────────────────────────────────────
PROMPT_ENGINEER_SYSTEM = """You are an expert AI image generation prompt engineer specializing in FLUX diffusion models. Your job is to evaluate and optimize user prompts for maximum image quality, brand consistency, and reference image fidelity.

You operate in two phases:

---

## PHASE 1: EVALUATION

Score the prompt across the following dimensions. Each dimension is scored 1–10. Be critical and precise.

### Scoring Dimensions

**1. Reference Anchoring (1–10)**
Does the prompt explicitly reference the provided input images by index (e.g., "image 1", "image 2")? Are the roles of each reference image (background, model, product) clearly stated? Vague or missing reference anchors cause FLUX to ignore or blend reference images incorrectly.

**2. Scene Specificity (1–10)**
Is the scene described with enough physical detail — location, time of day, weather, surface textures, spatial composition? Generic scene descriptions produce generic outputs. Penalize abstract or vague language ("nice background", "natural setting").

**3. Photographic Language (1–10)**
Does the prompt include camera and lighting descriptors? Look for: lens focal length, aperture, lighting type (e.g., softbox, golden hour, rim lighting), shot angle, depth of field. FLUX responds strongly to photographic vocabulary. Penalize prompts that read like creative writing rather than a cinematographer's brief.

**4. Subject Clarity (1–10)**
Is the primary subject — model, product, outfit — described unambiguously? Are pose, expression, gaze direction, and body framing specified? Penalize prompts where the subject's position in the scene is left to chance.

**5. Style Coherence (1–10)**
Are the style descriptors consistent and non-contradictory? Conflicting signals (e.g., "cinematic film grain" + "clean digital studio") degrade FLUX attention. Penalize mixed or redundant style tokens. Penalize low-signal filler terms ("ultra HD", "8K", "photorealistic", "masterpiece") — these have no effect on FLUX output resolution or quality.

**6. Brand & Commercial Readiness (1–10)**
Is the output likely to be polished enough for commercial use? Consider: clean composition, absence of surreal artifacts in the description, professional tone. Prompts that describe cluttered, chaotic, or stylistically experimental scenes score lower.

---

Output Phase 1 as structured JSON:

{
  "scores": {
    "reference_anchoring": <int>,
    "scene_specificity": <int>,
    "photographic_language": <int>,
    "subject_clarity": <int>,
    "style_coherence": <int>,
    "brand_commercial_readiness": <int>
  },
  "overall_score": <float, average of above>,
  "issues": [
    "<concise description of specific problem>",
    ...
  ],
  "strengths": [
    "<concise description of what is working>",
    ...
  ]
}

---

## PHASE 2: OPTIMIZATION

Rewrite the prompt to fix every issue identified in Phase 1. Follow these rules strictly:

### Rewrite Rules

1. **Always anchor reference images explicitly.**
   Use the format: "The model from image 2 is wearing the outfit from image 3, posed in the environment from image 1."
   Never leave reference roles implicit.

2. **Describe the scene photographically.**
   Include: focal length, aperture or depth of field, lighting setup, shot angle, time of day if outdoors.
   Example: "Shot on 85mm f/1.8 lens. Soft diffused studio lighting with a warm key light from camera left and a subtle rim light separating the subject from the background."

3. **Specify the subject completely.**
   State pose, expression, gaze direction, and framing (full body / waist-up / close-up). If the product is the hero, describe how it is held, placed, or worn.

4. **Use consistent, purposeful style language.**
   Choose one visual register and stay in it. Commercial editorial photography is the default. Remove all filler quality tokens (8K, ultra HD, hyperrealistic, masterpiece, best quality, etc.).

5. **Preserve the user's original creative intent.**
   Do not invent new concepts, locations, or characters that were not implied by the original prompt. Expand and sharpen — do not replace.

6. **Keep the optimized prompt under 120 words.**
   FLUX attention degrades with excessively long prompts. Prioritize density and precision over completeness.

---

Output Phase 2 as structured JSON:

{
  "optimized_prompt": "<rewritten prompt>",
  "changes_made": [
    "<description of specific change and why>",
    ...
  ],
  "score_delta": {
    "reference_anchoring": <int, change from original>,
    "scene_specificity": <int>,
    "photographic_language": <int>,
    "subject_clarity": <int>,
    "style_coherence": <int>,
    "brand_commercial_readiness": <int>
  }
}

---

## FINAL OUTPUT FORMAT

Return a single JSON object combining both phases:

{
  "evaluation": { ...Phase 1 output... },
  "optimization": { ...Phase 2 output... }
}

Return only valid JSON. No preamble, no markdown, no explanation outside the JSON structure.
Return a single JSON object combining both phases. Do not include any text, headers, or markdown outside the JSON. Do not split into two separate JSON objects. Your entire response must be one valid JSON object starting with {{ and ending with }}."""
"""

# ─────────────────────────────────────────────
# Global model handles
# ─────────────────────────────────────────────
flux_pipe   = None   # Klein only
vton_proc   = None
_upsampler  = None   # Real-ESRGAN (lazy-loaded)

# In-memory job store
jobs: dict = {}

# ─────────────────────────────────────────────
# AWS / S3
# ─────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4"),
)


def upload_pil_to_s3(img: Image.Image, key: str) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf, ContentType="image/png")
    return f"{CLOUDFRONT_OUTPUTS}/{key}"


# ─────────────────────────────────────────────
# Real-ESRGAN upscaler (lazy-loaded)
# ─────────────────────────────────────────────
def get_upsampler():
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise ImportError(
            "Real-ESRGAN not installed. Run: pip install basicsr realesrgan"
        )

    weights_path = Path("weights/RealESRGAN_x4plus.pth")
    if not weights_path.exists():
        log.info("Downloading RealESRGAN_x4plus weights...")
        weights_path.parent.mkdir(exist_ok=True)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        r = httpx.get(url, follow_redirects=True, timeout=120)
        weights_path.write_bytes(r.content)
        log.info("✓ RealESRGAN weights downloaded")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=4,
    )

    _upsampler = RealESRGANer(
        scale=4,
        model_path=str(weights_path),
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )
    log.info("✓ Real-ESRGAN upsampler ready")
    return _upsampler


def upscale_pil(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Run Real-ESRGAN 4× on a PIL Image. Returns a PIL Image.
    strength: 0.0 = original, 1.0 = full upscale, 0.5 = blend.
    """
    import cv2
    import numpy as np

    upsampler = get_upsampler()

    # PIL → BGR numpy (what cv2 / ESRGAN expects)
    img_np = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    upscaled_bgr, _ = upsampler.enhance(img_bgr, outscale=4)

    if strength < 1.0:
        orig_resized = cv2.resize(
            img_bgr,
            (upscaled_bgr.shape[1], upscaled_bgr.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
        blended = cv2.addWeighted(upscaled_bgr, strength, orig_resized, 1.0 - strength, 0)
        result_bgr = blended
    else:
        result_bgr = upscaled_bgr

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


# ─────────────────────────────────────────────
# Klein model loader
# ─────────────────────────────────────────────
def load_flux_klein():
    global flux_pipe
    if flux_pipe is not None:
        return

    import torch
    from diffusers import Flux2KleinPipeline
    from huggingface_hub import login

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(TORCH_DTYPE_STR, torch.bfloat16)

    log.info("Logging into Hugging Face...")
    login(token=HF_TOKEN)

    log.info(f"Loading FLUX.2-klein from {FLUX_REPO_ID}...")
    flux_pipe = Flux2KleinPipeline.from_pretrained(
        FLUX_REPO_ID,
        torch_dtype=torch_dtype,
        device_map="balanced",
    )
    log.info("✓ FLUX.2-klein ready")


def load_vton():
    global vton_proc
    try:
        from fashn_vton import TryOnPipeline
        log.info("Loading Fashn VTON 1.5...")
        vton_proc = TryOnPipeline(weights_dir="./weights", device="cuda:0")
        log.info("✓ VTON ready")
    except Exception as e:
        log.warning(f"VTON not available: {e}")
        vton_proc = None


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if KLEIN_ENABLED:
        load_flux_klein()
    load_vton()
    yield
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="Brand Style Generator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def pil_to_b64(img: Image.Image, max_size: int = 1024) -> str:
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def bytes_to_b64(data: bytes, max_size: int = 1024) -> str:
    return pil_to_b64(bytes_to_pil(data), max_size)


def fetch_image_from_url(url: str) -> Image.Image:
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ─────────────────────────────────────────────
# BFL API helpers
# ─────────────────────────────────────────────
def bfl_headers() -> dict:
    return {"X-Key": BFL_API_KEY, "Content-Type": "application/json"}


def bfl_submit(payload: dict, model: str) -> str:
    url = f"{BFL_BASE_URL}/{model}"
    resp = httpx.post(url, json=payload, headers=bfl_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("id")
    if not task_id:
        raise RuntimeError(f"BFL did not return a task id: {data}")
    log.info(f"BFL task submitted: {task_id} (model={model})")
    return task_id


def bfl_poll(task_id: str) -> str:
    poll_url = f"{BFL_BASE_URL}/get_result"
    deadline = time.time() + BFL_POLL_TIMEOUT

    while time.time() < deadline:
        resp = httpx.get(poll_url, params={"id": task_id}, headers=bfl_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("status", "")

        if state == "Ready":
            image_url = data["result"]["sample"]
            log.info(f"BFL task {task_id} ready")
            return image_url
        if state in ("Error", "Failed", "Content Moderated", "Request Moderated"):
            raise RuntimeError(f"BFL task {task_id} ended with status '{state}': {data}")

        log.debug(f"BFL task {task_id}: {state}")
        time.sleep(BFL_POLL_INTERVAL)

    raise TimeoutError(f"BFL task {task_id} did not complete within {BFL_POLL_TIMEOUT}s")


def ensure_image_references(prompt: str, ref_count: int) -> str:
    """Inject 'Referencing image N' prefix for any index not already in the prompt."""
    injections = []
    for n in range(1, ref_count + 1):
        if not re.search(rf"\bimage\s*{n}\b", prompt, re.IGNORECASE):
            injections.append(f"image {n}")
    if not injections:
        return prompt
    return f"Referencing {', '.join(injections)}. {prompt}"


def build_bfl_payload(
    prompt: str,
    flux_steps: int,
    guidance_scale: float,
    model_ref_bytes: Optional[bytes],
    bg_ref_bytes: Optional[bytes],
    additional_refs_bytes: List[bytes],
    width: int = 1024,
    height: int = 1024,
) -> dict:
    # Count active refs to inject anchors for any that are missing
    active_refs = [b for b in [bg_ref_bytes, model_ref_bytes, *additional_refs_bytes] if b]
    prompt = ensure_image_references(prompt, len(active_refs))

    payload: dict = {
        "prompt":           prompt,
        "width":            width,
        "height":           height,
        "steps":            flux_steps,
        "guidance":         guidance_scale,
        "output_format":    "png",
        "safety_tolerance": 2,
        "seed":             random.randint(0, 2**32 - 1),
    }

    # Primary image_prompt: model ref > bg ref
    primary_ref = model_ref_bytes or bg_ref_bytes
    if primary_ref:
        payload["image_prompt"]          = bytes_to_b64(primary_ref)
        payload["image_prompt_strength"] = 0.15

    # Verbal injection for secondary refs
    ref_notes: list[str] = []
    if bg_ref_bytes and model_ref_bytes:
        ref_notes.append("Use the location and lighting from the background reference.")
    if additional_refs_bytes:
        ref_notes.append(
            f"Incorporate style/product details from {len(additional_refs_bytes)} "
            "additional reference image(s)."
        )
    if ref_notes:
        payload["prompt"] = " ".join(ref_notes) + " " + prompt

    return payload


# ─────────────────────────────────────────────
# Core generation (background thread)
# ─────────────────────────────────────────────
def run_generation(
    job_id: str,
    prompt: str,
    model_choice: str,        # "klein" | "flux-2-pro" | "flux-2-max"
    use_vton: bool,
    use_upscale: bool,
    upscale_strength: float,
    model_ref_bytes: Optional[bytes],
    bg_ref_bytes: Optional[bytes],
    additional_refs_bytes: List[bytes],
    garment_bytes: Optional[bytes],
    garment_category: str,
    flux_steps: int,
    guidance_scale: float,
    vton_steps: int,
    width: int,
    height: int,
):
    try:
        jobs[job_id]["status"] = "running"
        flux_out: Image.Image

        # ── Klein (local inference) ──────────────────────────────────
        if model_choice == "klein":
            if flux_pipe is None:
                raise RuntimeError("Klein model not loaded. Set KLEIN_ENABLED=true and restart.")

            import torch
            generator = torch.Generator(device="cuda:0").manual_seed(random.randint(0, 2**32))

            refs = []
            if bg_ref_bytes:
                refs.append(bytes_to_pil(bg_ref_bytes))
            if model_ref_bytes:
                refs.append(bytes_to_pil(model_ref_bytes))
            for b in additional_refs_bytes:
                refs.append(bytes_to_pil(b))

            log.info(f"[{job_id}] Klein inference — {len(refs)} refs, steps={flux_steps}")
            if refs:
                flux_out = flux_pipe(
                    prompt=prompt,
                    image=refs,
                    generator=generator,
                    num_inference_steps=flux_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            else:
                flux_out = flux_pipe(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=flux_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                ).images[0]

        # ── BFL API (Pro / Max) ──────────────────────────────────────
        else:
            if not BFL_API_KEY:
                raise RuntimeError("BFL_API_KEY not configured")

            payload = build_bfl_payload(
                prompt=prompt,
                flux_steps=flux_steps,
                guidance_scale=guidance_scale,
                model_ref_bytes=model_ref_bytes,
                bg_ref_bytes=bg_ref_bytes,
                additional_refs_bytes=additional_refs_bytes,
                width=width,
                height=height,
            )
            log.info(f"[{job_id}] BFL submit — model={model_choice}, VTON={use_vton}")
            task_id   = bfl_submit(payload, model=model_choice)
            image_url = bfl_poll(task_id)
            flux_out  = fetch_image_from_url(image_url)

        final_out     = flux_out
        upscaled_out  = None

        # ── Real-ESRGAN upscaling ────────────────────────────────────
        if use_upscale:
            log.info(f"[{job_id}] Upscaling (strength={upscale_strength})...")
            upscaled_out = upscale_pil(flux_out, strength=upscale_strength)
            log.info(f"[{job_id}] ✓ Upscale complete")

        # ── VTON (optional, always local) ────────────────────────────
        if use_vton and vton_proc is not None and garment_bytes:
            log.info(f"[{job_id}] Applying VTON...")
            source = upscaled_out if upscaled_out is not None else flux_out
            garment_img = bytes_to_pil(garment_bytes)
            final_out = vton_proc(
                person_image=source,
                garment_image=garment_img,
                category=garment_category,
            ).images[0]
        elif use_vton and vton_proc is None:
            log.warning(f"[{job_id}] VTON requested but not loaded — skipping")

        # ── Upload to S3 ─────────────────────────────────────────────
        flux_url     = upload_pil_to_s3(flux_out,  f"outputs/{job_id}/flux_output.png")
        upscaled_url = upload_pil_to_s3(upscaled_out, f"outputs/{job_id}/upscaled_output.png") if upscaled_out else None
        final_url    = upload_pil_to_s3(final_out, f"outputs/{job_id}/final_output.png")

        jobs[job_id].update({
            "status":         "done",
            "model":          model_choice,
            "flux_url":       flux_url,
            "upscaled_url":   upscaled_url,
            "final_url":      final_url,
            "upscale_applied": upscaled_out is not None,
            "vton_applied":   use_vton and vton_proc is not None and garment_bytes is not None,
        })
        log.info(f"[{job_id}] ✓ Complete (model={model_choice})")

    except Exception as e:
        log.exception(f"[{job_id}] Generation failed")
        jobs[job_id].update({"status": "error", "error": str(e)})
    finally:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        import torch
        gpu_mem = round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0
        device  = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        gpu_mem = 0
        device  = "cpu"

    return {
        "status":        "ok",
        "device":        device,
        "klein_enabled": KLEIN_ENABLED,
        "klein_loaded":  flux_pipe is not None,
        "vton_loaded":   vton_proc is not None,
        "bfl_key_set":   bool(BFL_API_KEY),
        "ollama_model":  OLLAMA_MODEL,
        "gpu_memory_gb": gpu_mem,
    }


@app.post("/generate")
async def generate(
    prompt:               str   = Form(...),
    model_choice:         str   = Form("flux-2-pro"),   # "klein" | "flux-2-pro" | "flux-2-max"
    use_vton:             bool  = Form(False),
    use_upscale:          bool  = Form(False),
    upscale_strength:     float = Form(1.0),            # 0.0–1.0
    garment_category:     str   = Form("tops"),
    flux_steps:           int   = Form(28),
    guidance_scale:       float = Form(3.5),
    vton_steps:           int   = Form(30),
    width:                int   = Form(1024),
    height:               int   = Form(1024),
    model_reference:      Optional[UploadFile] = File(None),
    background_reference: Optional[UploadFile] = File(None),
    garment_image:        Optional[UploadFile] = File(None),
    additional_ref_1:     Optional[UploadFile] = File(None),
    additional_ref_2:     Optional[UploadFile] = File(None),
    _key: str = Security(verify_key),
):
    """Submit a generation job. Returns job_id immediately — poll /status/{job_id}."""
    valid_models = {"klein", "flux-2-pro", "flux-2-max"}
    if model_choice not in valid_models:
        raise HTTPException(400, f"model_choice must be one of {valid_models}")

    if model_choice == "klein" and flux_pipe is None:
        raise HTTPException(503, "Klein model not loaded. Start the server with KLEIN_ENABLED=true.")

    if model_choice in ("flux-2-pro", "flux-2-max") and not BFL_API_KEY:
        raise HTTPException(503, "BFL_API_KEY not configured")

    upscale_strength = max(0.0, min(1.0, upscale_strength))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": time.time(), "model": model_choice}

    model_bytes   = await model_reference.read()      if model_reference      else None
    bg_bytes      = await background_reference.read() if background_reference else None
    garment_bytes = await garment_image.read()        if garment_image        else None
    add1          = await additional_ref_1.read()     if additional_ref_1     else None
    add2          = await additional_ref_2.read()     if additional_ref_2     else None
    additional    = [b for b in [add1, add2] if b]

    thread = threading.Thread(
        target=run_generation,
        args=(
            job_id, prompt, model_choice,
            use_vton, use_upscale, upscale_strength,
            model_bytes, bg_bytes, additional,
            garment_bytes, garment_category,
            flux_steps, guidance_scale, vton_steps,
            width, height,
        ),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued", "model": model_choice}


@app.get("/status/{job_id}")
def status(job_id: str, _key: str = Security(verify_key)):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, _key: str = Security(verify_key)):
    jobs.pop(job_id, None)
    return {"deleted": job_id}


# ─────────────────────────────────────────────
# Prompt improvement via Ollama
# ─────────────────────────────────────────────

@app.post("/improve-prompt")
async def improve_prompt(
    prompt: str = Form(...),
    _key: str = Security(verify_key),
):
    """
    Run the prompt through the local Ollama model for evaluation + optimization.
    Returns the full JSON from the prompt engineer system prompt.
    """
    if not prompt.strip():
        raise HTTPException(400, "prompt is empty")

    ollama_url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_ENGINEER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,   # low temp for consistent structured output
            "num_predict": 2048,
        },
    }

    try:
        resp = httpx.post(ollama_url, json=payload, timeout=120)
        resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, f"Ollama not reachable at {OLLAMA_BASE_URL}. Is it running?")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Ollama error: {e.response.text}")

    data = resp.json()
    raw_text = data.get("message", {}).get("content", "")

    # Strip markdown fences if the model wrapped the JSON
    clean = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()

    # Extract the outermost JSON object — handles models that prepend/append text
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if not match:
        raise HTTPException(500, f"No JSON object found in Ollama response: {raw_text[:500]}")

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"Failed to parse JSON from Ollama response: {exc} — raw: {raw_text[:500]}")

    return result


@app.get("/ollama/models")
async def list_ollama_models(_key: str = Security(verify_key)):
    """List models available in the local Ollama instance."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(503, f"Could not reach Ollama: {e}")