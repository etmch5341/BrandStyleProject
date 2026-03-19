"""
LoRA Training Script for SDXL
Trains a LoRA adapter on custom image-caption pairs
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import json
from datetime import datetime


class ImageCaptionDataset(Dataset):
    """Dataset for loading images and their corresponding captions from txt files"""
    
    def __init__(self, data_dir, tokenizer, tokenizer_2, size=1024):
        """
        Args:
            data_dir: Directory containing images and .txt caption files
            tokenizer: CLIP tokenizer for text encoder 1
            tokenizer_2: CLIP tokenizer for text encoder 2
            size: Target image size (default 1024 for SDXL)
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        
        # Find all images with corresponding caption files
        self.image_paths = []
        self.caption_paths = []
        
        # Support common image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        
        for ext in image_extensions:
            for img_path in self.data_dir.glob(f'*{ext}'):
                caption_path = img_path.with_suffix('.txt')
                if caption_path.exists():
                    self.image_paths.append(img_path)
                    self.caption_paths.append(caption_path)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No image-caption pairs found in {data_dir}")
        
        print(f"Found {len(self.image_paths)} image-caption pairs")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        
        # Load caption
        with open(self.caption_paths[idx], 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        # Tokenize caption for both text encoders
        tokens_1 = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        tokens_2 = self.tokenizer_2(
            caption,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            'pixel_values': image,
            'input_ids_1': tokens_1,
            'input_ids_2': tokens_2,
            'caption': caption
        }


def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids_1 = torch.stack([example["input_ids_1"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "input_ids_1": input_ids_1,
        "input_ids_2": input_ids_2,
    }


def encode_prompt(text_encoder, text_encoder_2, input_ids_1, input_ids_2):
    """Encode prompts using both CLIP text encoders"""
    # Encode with first text encoder
    prompt_embeds_1 = text_encoder(input_ids_1, output_hidden_states=True)
    prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
    
    # Encode with second text encoder  
    prompt_embeds_2 = text_encoder_2(input_ids_2, output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds_2[0]  # Get pooled embeddings from text_encoder_2
    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
    
    # Concatenate embeddings
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    
    return prompt_embeds, pooled_prompt_embeds


def train_lora(args):
    """Main training function"""
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    # Load models
    print("Loading models...")
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    vae.requires_grad_(False)
    
    # Text Encoders
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    text_encoder.requires_grad_(False)
    
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    text_encoder_2.requires_grad_(False)
    
    # Tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2"
    )
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    # Add LoRA layers to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    # Dataset and DataLoader
    print("Loading dataset...")
    dataset = ImageCaptionDataset(
        args.data_dir,
        tokenizer,
        tokenizer_2,
        size=args.resolution
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Learning rate scheduler
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    
    # Training loop
    print("Starting training...")
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoder,
                        text_encoder_2,
                        batch["input_ids_1"],
                        batch["input_ids_2"]
                    )
                
                # Add time embeddings
                add_time_ids = torch.cat([
                    torch.tensor([[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]]) 
                    for _ in range(bsz)
                ]).to(device=latents.device, dtype=latents.dtype)
                
                # Prepare added conditions
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds.to(dtype=latents.dtype),
                    "time_ids": add_time_ids
                }
                
                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                
                # Log progress
                if global_step % args.log_interval == 0:
                    avg_loss = train_loss / args.log_interval
                    progress_bar.set_postfix({"loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]})
                    train_loss = 0.0
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unet.save_pretrained(save_path)
                        print(f"Saved checkpoint to {save_path}")
            
            if global_step >= args.max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "lora_final")
        unet.save_pretrained(final_path)
        
        # Save training config
        config = {
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "base_model": args.pretrained_model_name_or_path,
            "training_steps": global_step,
            "learning_rate": args.learning_rate,
            "trained_on": datetime.now().isoformat()
        }
        with open(os.path.join(final_path, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Training complete! Model saved to {final_path}")
    
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for SDXL on custom dataset")
    
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing images and .txt caption files",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution for training",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank (higher = more capacity, more VRAM)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor)",
    )
    
    # Training arguments
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler",
    )
    
    # Optimizer arguments
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    
    # Other arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    train_lora(args)