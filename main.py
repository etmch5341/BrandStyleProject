"""
Brand Style Marketing Generator — FastAPI Backend
Converted from FLUX.2-klein + VTON v1.5 notebook
"""

import os
import gc
import uuid
import time
import threading
import random
import logging
from io import BytesIO
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import boto3
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("brand-gen")

# ─────────────────────────────────────────────
# Config  (set via environment variables on EC2)
# ─────────────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")
API_KEY         = os.getenv("API_KEY", "changeme-secret-key")
S3_BUCKET       = os.getenv("S3_BUCKET", "your-brand-gen-outputs")
AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")
FLUX_REPO_ID    = "black-forest-labs/FLUX.2-klein-4B"
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE     = torch.bfloat16

# ─────────────────────────────────────────────
# Global model handles
# ─────────────────────────────────────────────
flux_pipe   = None
vton_proc   = None

# In-memory job store  { job_id: { status, result_url, error, created_at } }
jobs: dict = {}

# ─────────────────────────────────────────────
# S3 client
# ─────────────────────────────────────────────
s3 = boto3.client("s3", region_name=AWS_REGION)


def upload_pil_to_s3(img: Image.Image, key: str) -> str:
    """Save PIL image to S3 and return a presigned URL (1 hour)."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf, ContentType="image/png")
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=3600,
    )
    return url


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_flux():
    global flux_pipe
    from diffusers import Flux2KleinPipeline
    from huggingface_hub import login

    log.info("Logging into Hugging Face...")
    login(token=HF_TOKEN)

    log.info(f"Loading FLUX.2-klein from {FLUX_REPO_ID} ...")
    flux_pipe = Flux2KleinPipeline.from_pretrained(
        FLUX_REPO_ID, torch_dtype=TORCH_DTYPE
    ).to(DEVICE)
    flux_pipe.enable_model_cpu_offload()
    log.info("✓ FLUX.2-klein ready")


def load_vton():
    global vton_proc
    try:
        from fashn_vton import TryOnPipeline
        log.info("Loading Fashn VTON 1.5...")
        vton_proc = TryOnPipeline(weights_dir="./weights", device=DEVICE)
        log.info("✓ VTON ready")
    except Exception as e:
        log.warning(f"VTON not available: {e}")
        vton_proc = None


# ─────────────────────────────────────────────
# Lifespan  — load models once on startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_flux()
    load_vton()
    yield
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="Brand Style Generator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Lock this down to your CloudFront domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ─────────────────────────────────────────────
# Helpers — load image from bytes
# ─────────────────────────────────────────────
def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


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
        generator = torch.Generator(device=DEVICE).manual_seed(random.randint(0, 2**32))

        # ── Build reference list: Location → Model → Additional ──
        refs = []
        if bg_ref_bytes:
            refs.append(bytes_to_pil(bg_ref_bytes))
        if model_ref_bytes:
            refs.append(bytes_to_pil(model_ref_bytes))
        for b in additional_refs_bytes:
            refs.append(bytes_to_pil(b))

        # ── FLUX generation ──
        log.info(f"[{job_id}] Running FLUX with {len(refs)} references, VTON={use_vton}")
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
                width=1024,
                height=1024,
            ).images[0]

        final_out = flux_out

        # ── VTON (optional) ──
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
            "status":    "done",
            "flux_url":  flux_url,
            "final_url": final_url,
            "vton_applied": use_vton and vton_proc is not None and garment_bytes is not None,
        })
        log.info(f"[{job_id}] ✓ Complete")

    except Exception as e:
        log.exception(f"[{job_id}] Generation failed")
        jobs[job_id].update({"status": "error", "error": str(e)})
    finally:
        torch.cuda.empty_cache()
        gc.collect()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "flux_loaded": flux_pipe is not None,
        "vton_loaded": vton_proc is not None,
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
    }


@app.post("/generate")
async def generate(
    prompt:             str         = Form(...),
    use_vton:           bool        = Form(False),
    garment_category:   str         = Form("tops"),
    flux_steps:         int         = Form(4),
    guidance_scale:     float       = Form(4.0),
    vton_steps:         int         = Form(30),
    model_reference:    Optional[UploadFile] = File(None),
    background_reference: Optional[UploadFile] = File(None),
    garment_image:      Optional[UploadFile] = File(None),
    additional_ref_1:   Optional[UploadFile] = File(None),
    additional_ref_2:   Optional[UploadFile] = File(None),
    _key: str = Security(verify_key),
):
    """
    Submit a generation job. Returns a job_id immediately.
    Poll /status/{job_id} for progress.
    """
    if flux_pipe is None:
        raise HTTPException(503, "Model not loaded yet")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": time.time()}

    # Read file bytes (FastAPI UploadFile is async)
    model_bytes   = await model_reference.read()   if model_reference   else None
    bg_bytes      = await background_reference.read() if background_reference else None
    garment_bytes = await garment_image.read()     if garment_image     else None
    add1          = await additional_ref_1.read()  if additional_ref_1  else None
    add2          = await additional_ref_2.read()  if additional_ref_2  else None
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
    """Poll this endpoint to check job progress."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, _key: str = Security(verify_key)):
    """Clean up a completed job record."""
    jobs.pop(job_id, None)
    return {"deleted": job_id}
