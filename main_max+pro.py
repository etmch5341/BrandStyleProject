"""
Brand Style Marketing Generator — FastAPI Backend
Uses BFL FLUX.2 Pro/Max API instead of local model inference
"""

import os
import gc
import re
import uuid
import time
import base64
import threading
import random
import logging
import httpx
from io import BytesIO
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
# Config  (set via environment variables on EC2)
# ─────────────────────────────────────────────
API_KEY             = os.getenv("API_KEY", "changeme-secret-key")
S3_BUCKET           = os.getenv("S3_BUCKET", "your-brand-gen-outputs")
AWS_REGION          = os.getenv("AWS_REGION", "us-east-2")
BFL_API_KEY         = os.getenv("BFL_API_KEY", "")

# Switch between "flux-pro-1.1-ultra" and "flux-2-pro" / "flux-2-max"
BFL_MODEL           = os.getenv("BFL_MODEL", "flux-2-pro")
BFL_BASE_URL        = "https://api.bfl.ai/v1"

# How long (seconds) to wait for BFL to finish before giving up
BFL_POLL_TIMEOUT    = int(os.getenv("BFL_POLL_TIMEOUT", "300"))
BFL_POLL_INTERVAL   = float(os.getenv("BFL_POLL_INTERVAL", "2.0"))

# ─────────────────────────────────────────────
# VTON global handle (still runs locally)
# ─────────────────────────────────────────────
vton_proc = None

# In-memory job store  { job_id: { status, result_url, error, created_at } }
jobs: dict = {}

# ─────────────────────────────────────────────
# S3 + CloudFront
# ─────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4"),
)

CLOUDFRONT_OUTPUTS = "https://d52unpfog765b.cloudfront.net"


def upload_pil_to_s3(img: Image.Image, key: str) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf, ContentType="image/png")
    return f"{CLOUDFRONT_OUTPUTS}/{key}"


# ─────────────────────────────────────────────
# VTON loader (optional — unchanged from before)
# ─────────────────────────────────────────────
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
# Lifespan — only VTON loads locally now
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vton()
    yield


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
    """Resize if needed, return base64-encoded PNG string."""
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def bytes_to_b64(data: bytes, max_size: int = 1024) -> str:
    return pil_to_b64(bytes_to_pil(data), max_size)


def fetch_image_from_url(url: str) -> Image.Image:
    """Download BFL result image and return as PIL Image."""
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ─────────────────────────────────────────────
# BFL API helpers
# ─────────────────────────────────────────────
def bfl_headers() -> dict:
    return {
        "X-Key": BFL_API_KEY,
        "Content-Type": "application/json",
    }


def bfl_submit(payload: dict) -> str:
    """
    POST to BFL endpoint and return the task ID.
    Raises on HTTP error.
    """
    url = f"{BFL_BASE_URL}/{BFL_MODEL}"
    resp = httpx.post(url, json=payload, headers=bfl_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("id")
    if not task_id:
        raise RuntimeError(f"BFL did not return a task id: {data}")
    log.info(f"BFL task submitted: {task_id}")
    return task_id


def bfl_poll(task_id: str) -> str:
    """
    Poll BFL's get_result endpoint until the task is Ready.
    Returns the final image URL.
    Raises on timeout or error status.
    """
    poll_url = f"{BFL_BASE_URL}/get_result"
    deadline = time.time() + BFL_POLL_TIMEOUT

    while time.time() < deadline:
        resp = httpx.get(
            poll_url,
            params={"id": task_id},
            headers=bfl_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        state = data.get("status", "")

        if state == "Ready":
            image_url = data["result"]["sample"]
            log.info(f"BFL task {task_id} ready: {image_url}")
            return image_url

        if state in ("Error", "Failed", "Content Moderated", "Request Moderated"):
            raise RuntimeError(f"BFL task {task_id} ended with status '{state}': {data}")

        log.debug(f"BFL task {task_id} status: {state} — waiting...")
        time.sleep(BFL_POLL_INTERVAL)

    raise TimeoutError(f"BFL task {task_id} did not complete within {BFL_POLL_TIMEOUT}s")


def ensure_image_references(prompt: str, refs: list) -> str:
    """
    For each reference image in the ordered list, check that the prompt
    already mentions it (e.g. 'image 1', 'image1', 'Image 1').
    Injects a minimal prefix only for any that are missing, leaving
    well-formed prompts completely unchanged.
    """
    injections = []
    for i in range(len(refs)):
        n = i + 1
        if not re.search(rf"\bimage\s*{n}\b", prompt, re.IGNORECASE):
            injections.append(f"image {n}")

    if not injections:
        return prompt

    ref_str = ", ".join(injections)
    return f"Referencing {ref_str}. {prompt}"


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
    """
    Build the BFL API request payload.

    Reference image strategy
    ────────────────────────
    The BFL API (flux-pro-1.1-ultra and expected FLUX.2 endpoints) supports a
    single `image_prompt` field (base64) plus `image_prompt_strength` [0–1].
    True ordered multi-reference conditioning is only possible with local
    Diffusers; via the API we approximate it:

      • image_prompt  → model reference (strongest identity anchor)
      • bg / additional refs → described verbally in an injected prompt prefix

    If no model reference is provided, we fall back to the background reference
    as image_prompt. This gives the model the most visually grounding signal
    while keeping everything within the API contract.
    """
    prompt = ensure_image_references(prompt, refs)

    payload: dict = {
        "prompt":          prompt,
        "width":           width,
        "height":          height,
        "steps":           flux_steps,
        "guidance":        guidance_scale,
        "output_format":   "png",
        "safety_tolerance": 2,
        "seed":            random.randint(0, 2**32 - 1),
    }

    # ── Pick the primary image_prompt ──
    primary_ref = model_ref_bytes or bg_ref_bytes
    if primary_ref:
        payload["image_prompt"]          = bytes_to_b64(primary_ref)
        payload["image_prompt_strength"] = 0.15  # subtle conditioning; tune as needed

    # ── Inject a verbal reference block for remaining images ──
    # This replaces what the Diffusers pipeline handled via cross-attention on
    # the ordered refs list. Keep it compact — FLUX reads prompt tokens well.
    ref_notes: list[str] = []
    if bg_ref_bytes and model_ref_bytes:
        # Both present: model is image_prompt, describe bg verbally
        ref_notes.append("Use the location and lighting from the background reference.")
    elif bg_ref_bytes and not model_ref_bytes:
        # bg became image_prompt; no extra note needed for it
        pass

    if additional_refs_bytes:
        ref_notes.append(
            f"Incorporate style/product details from {len(additional_refs_bytes)} additional reference image(s)."
        )

    if ref_notes:
        injected = " ".join(ref_notes)
        payload["prompt"] = f"{injected} {prompt}"

    return payload


# ─────────────────────────────────────────────
# Core generation  (runs in background thread)
# ─────────────────────────────────────────────
def run_generation(
    job_id: str,
    prompt: str,
    use_vton: bool,
    model_ref_bytes: Optional[bytes],
    bg_ref_bytes: Optional[bytes],
    additional_refs_bytes: List[bytes],
    garment_bytes: Optional[bytes],
    garment_category: str,
    flux_steps: int,
    guidance_scale: float,
    vton_steps: int,
):
    try:
        jobs[job_id]["status"] = "running"

        # ── Build and submit BFL request ──
        payload = build_bfl_payload(
            prompt=prompt,
            flux_steps=flux_steps,
            guidance_scale=guidance_scale,
            model_ref_bytes=model_ref_bytes,
            bg_ref_bytes=bg_ref_bytes,
            additional_refs_bytes=additional_refs_bytes,
        )

        log.info(f"[{job_id}] Submitting to BFL model={BFL_MODEL}, VTON={use_vton}")
        task_id   = bfl_submit(payload)
        image_url = bfl_poll(task_id)

        # ── Download result image ──
        flux_out  = fetch_image_from_url(image_url)
        final_out = flux_out

        # ── VTON (optional, still local) ──
        if use_vton and vton_proc is not None and garment_bytes:
            log.info(f"[{job_id}] Applying VTON...")
            garment_img = bytes_to_pil(garment_bytes)
            final_out = vton_proc(
                person_image=flux_out,
                garment_image=garment_img,
                category=garment_category,
            ).images[0]
        elif use_vton and vton_proc is None:
            log.warning(f"[{job_id}] VTON requested but model not loaded — skipping")

        # ── Upload both outputs to S3 ──
        flux_url  = upload_pil_to_s3(flux_out,  f"outputs/{job_id}/flux_output.png")
        final_url = upload_pil_to_s3(final_out, f"outputs/{job_id}/final_output.png")

        jobs[job_id].update({
            "status":       "done",
            "flux_url":     flux_url,
            "final_url":    final_url,
            "bfl_task_id":  task_id,
            "vton_applied": use_vton and vton_proc is not None and garment_bytes is not None,
        })
        log.info(f"[{job_id}] ✓ Complete")

    except Exception as e:
        log.exception(f"[{job_id}] Generation failed")
        jobs[job_id].update({"status": "error", "error": str(e)})


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "bfl_model":    BFL_MODEL,
        "vton_loaded":  vton_proc is not None,
        "bfl_key_set":  bool(BFL_API_KEY),
    }


@app.post("/generate")
async def generate(
    prompt:               str          = Form(...),
    use_vton:             bool         = Form(False),
    garment_category:     str          = Form("tops"),
    flux_steps:           int          = Form(28),    # BFL default ~28; Klein used 4
    guidance_scale:       float        = Form(3.5),
    vton_steps:           int          = Form(30),
    model_reference:      Optional[UploadFile] = File(None),
    background_reference: Optional[UploadFile] = File(None),
    garment_image:        Optional[UploadFile] = File(None),
    additional_ref_1:     Optional[UploadFile] = File(None),
    additional_ref_2:     Optional[UploadFile] = File(None),
    _key: str = Security(verify_key),
):
    """
    Submit a generation job. Returns a job_id immediately.
    Poll /status/{job_id} for progress.
    """
    if not BFL_API_KEY:
        raise HTTPException(503, "BFL_API_KEY not configured")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": time.time()}

    model_bytes   = await model_reference.read()      if model_reference      else None
    bg_bytes      = await background_reference.read() if background_reference else None
    garment_bytes = await garment_image.read()        if garment_image        else None
    add1          = await additional_ref_1.read()     if additional_ref_1     else None
    add2          = await additional_ref_2.read()     if additional_ref_2     else None
    additional    = [b for b in [add1, add2] if b]

    thread = threading.Thread(
        target=run_generation,
        args=(
            job_id, prompt, use_vton,
            model_bytes, bg_bytes, additional,
            garment_bytes, garment_category,
            flux_steps, guidance_scale, vton_steps,
        ),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str, _key: str = Security(verify_key)):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, _key: str = Security(verify_key)):
    jobs.pop(job_id, None)
    return {"deleted": job_id}