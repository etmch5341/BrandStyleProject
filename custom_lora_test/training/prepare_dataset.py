"""
Dataset Preparation and Validation Script
Helps prepare and validate image-caption datasets for LoRA training
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict


def validate_dataset(data_dir, min_resolution=512):
    """
    Validate dataset and report issues
    
    Args:
        data_dir: Directory containing images and captions
        min_resolution: Minimum acceptable image resolution
    """
    data_path = Path(data_dir)
    
    print(f"\n{'='*60}")
    print(f"Validating dataset: {data_dir}")
    print(f"{'='*60}\n")
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in data_path.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(images)} image files")
    
    # Check for captions
    valid_pairs = []
    missing_captions = []
    empty_captions = []
    invalid_images = []
    resolution_issues = []
    
    for img_path in images:
        caption_path = img_path.with_suffix('.txt')
        
        # Check caption exists
        if not caption_path.exists():
            missing_captions.append(img_path.name)
            continue
        
        # Check caption not empty
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        if not caption:
            empty_captions.append(img_path.name)
            continue
        
        # Check image validity
        try:
            img = Image.open(img_path)
            width, height = img.size
            
            # Check resolution
            if width < min_resolution or height < min_resolution:
                resolution_issues.append(f"{img_path.name} ({width}x{height})")
            
            valid_pairs.append({
                'image': img_path.name,
                'caption': caption,
                'resolution': f"{width}x{height}",
                'format': img.format
            })
            
        except Exception as e:
            invalid_images.append(f"{img_path.name} ({str(e)})")
    
    # Print report
    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT")
    print(f"{'='*60}\n")
    
    print(f"✓ Valid image-caption pairs: {len(valid_pairs)}")
    
    if missing_captions:
        print(f"\n✗ Missing caption files ({len(missing_captions)}):")
        for img in missing_captions[:10]:
            print(f"  - {img}")
        if len(missing_captions) > 10:
            print(f"  ... and {len(missing_captions) - 10} more")
    
    if empty_captions:
        print(f"\n✗ Empty caption files ({len(empty_captions)}):")
        for img in empty_captions[:10]:
            print(f"  - {img}")
        if len(empty_captions) > 10:
            print(f"  ... and {len(empty_captions) - 10} more")
    
    if invalid_images:
        print(f"\n✗ Invalid/corrupted images ({len(invalid_images)}):")
        for img in invalid_images[:10]:
            print(f"  - {img}")
        if len(invalid_images) > 10:
            print(f"  ... and {len(invalid_images) - 10} more")
    
    if resolution_issues:
        print(f"\n⚠ Low resolution images ({len(resolution_issues)}):")
        print(f"  (below {min_resolution}x{min_resolution})")
        for img in resolution_issues[:10]:
            print(f"  - {img}")
        if len(resolution_issues) > 10:
            print(f"  ... and {len(resolution_issues) - 10} more")
    
    # Statistics
    if valid_pairs:
        print(f"\n{'='*60}")
        print(f"DATASET STATISTICS")
        print(f"{'='*60}\n")
        
        # Caption length stats
        caption_lengths = [len(pair['caption'].split()) for pair in valid_pairs]
        avg_length = sum(caption_lengths) / len(caption_lengths)
        print(f"Caption length (words):")
        print(f"  - Average: {avg_length:.1f}")
        print(f"  - Min: {min(caption_lengths)}")
        print(f"  - Max: {max(caption_lengths)}")
        
        # Resolution stats
        resolutions = defaultdict(int)
        for pair in valid_pairs:
            resolutions[pair['resolution']] += 1
        
        print(f"\nImage resolutions:")
        for res, count in sorted(resolutions.items(), key=lambda x: -x[1])[:5]:
            print(f"  - {res}: {count} images")
        
        # Format stats
        formats = defaultdict(int)
        for pair in valid_pairs:
            formats[pair['format']] += 1
        
        print(f"\nImage formats:")
        for fmt, count in sorted(formats.items(), key=lambda x: -x[1]):
            print(f"  - {fmt}: {count} images")
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    if len(valid_pairs) == 0:
        print("✗ No valid training pairs found!")
        print("  Please add images with matching .txt caption files")
    elif len(valid_pairs) < 10:
        print("⚠ Very small dataset (<10 images)")
        print("  Consider adding more images for better results")
    elif len(valid_pairs) < 20:
        print("⚠ Small dataset (10-20 images)")
        print("  Recommend 20+ images for style/brand training")
    else:
        print("✓ Good dataset size!")
        print(f"  {len(valid_pairs)} valid pairs should train well")
    
    if resolution_issues:
        print(f"\n⚠ Some images below recommended {min_resolution}x{min_resolution} resolution")
        print("  Consider upscaling or removing low-res images")
    
    print(f"\n{'='*60}\n")
    
    return len(valid_pairs) > 0


def create_template_captions(data_dir, template=None):
    """
    Create template caption files for images that don't have them
    
    Args:
        data_dir: Directory containing images
        template: Caption template (use {filename} as placeholder)
    """
    data_path = Path(data_dir)
    
    if template is None:
        template = "a photo of {filename}"
    
    # Find images without captions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in data_path.iterdir() if f.suffix.lower() in image_extensions]
    
    created = 0
    for img_path in images:
        caption_path = img_path.with_suffix('.txt')
        
        if not caption_path.exists():
            # Create caption from template
            filename_no_ext = img_path.stem
            caption = template.replace('{filename}', filename_no_ext)
            
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            created += 1
            print(f"Created: {caption_path.name}")
    
    print(f"\nCreated {created} caption files")


def analyze_captions(data_dir):
    """Analyze caption content and suggest improvements"""
    data_path = Path(data_dir)
    
    # Find all caption files
    captions = []
    for txt_file in data_path.glob('*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
            if caption:
                captions.append(caption)
    
    if not captions:
        print("No captions found")
        return
    
    print(f"\n{'='*60}")
    print(f"CAPTION ANALYSIS")
    print(f"{'='*60}\n")
    
    # Word frequency
    from collections import Counter
    all_words = []
    for caption in captions:
        all_words.extend(caption.lower().split())
    
    word_freq = Counter(all_words)
    
    print(f"Total captions: {len(captions)}")
    print(f"Unique words: {len(word_freq)}")
    
    print(f"\nMost common words:")
    for word, count in word_freq.most_common(20):
        print(f"  {word}: {count}")
    
    # Check for potential trigger words
    print(f"\n{'='*60}")
    print(f"Potential trigger words (words appearing in >50% of captions):")
    threshold = len(captions) * 0.5
    trigger_candidates = [word for word, count in word_freq.items() if count >= threshold]
    
    if trigger_candidates:
        for word in trigger_candidates[:10]:
            print(f"  - {word} (in {word_freq[word]}/{len(captions)} captions)")
    else:
        print("  None found - consider adding consistent keywords")


def main():
    parser = argparse.ArgumentParser(description="Dataset preparation and validation")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing images and captions"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["validate", "create_captions", "analyze"],
        default="validate",
        help="Action to perform"
    )
    parser.add_argument(
        "--caption_template",
        type=str,
        default="a photo of {filename}",
        help="Template for creating captions (use {filename} as placeholder)"
    )
    parser.add_argument(
        "--min_resolution",
        type=int,
        default=512,
        help="Minimum acceptable image resolution"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist")
        return
    
    if args.action == "validate":
        validate_dataset(args.data_dir, args.min_resolution)
    elif args.action == "create_captions":
        create_template_captions(args.data_dir, args.caption_template)
    elif args.action == "analyze":
        analyze_captions(args.data_dir)


if __name__ == "__main__":
    main()