import argparse
from PIL import Image, ImageDraw, ImageFont
import os


def create_side_by_side_comparison(input_image, comparison_dir, output_path="comparison_grid.png"):
    """
    Create a side-by-side comparison grid of all model outputs
    """
    print("Creating visual comparison grid...\n")
    
    # Load original input
    input_img = Image.open(input_image).convert('RGB')
    
    # Upscale input to match output size (2x) for fair comparison
    w, h = input_img.size
    input_upscaled = input_img.resize((w*2, h*2), Image.BICUBIC)
    
    # Model outputs to compare
    models = [
        ("Input (Bicubic 2x)", input_upscaled),
        ("Lightweight", f"{comparison_dir}/output_lightweight.png"),
        ("Lightweight Quantized", f"{comparison_dir}/output_lightweight_quantized.png"),
        ("Improved", f"{comparison_dir}/output_improved.png"),
        ("Improved Quantized", f"{comparison_dir}/output_improved_quantized.png"),
    ]
    
    # Load all images
    images = []
    labels = []
    for label, path in models:
        if isinstance(path, str):
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                images.append(img)
                labels.append(label)
                print(f"✓ Loaded: {label}")
            else:
                print(f"⚠️  Skipped: {label} (not found)")
        else:
            images.append(path)
            labels.append(label)
            print(f"✓ Added: {label}")
    
    if not images:
        print("❌ No images to compare!")
        return
    
    # Create grid (2 columns)
    img_width, img_height = images[0].size
    cols = 2
    rows = (len(images) + cols - 1) // cols
    
    # Add space for labels
    label_height = 40
    grid_width = img_width * cols
    grid_height = (img_height + label_height) * rows
    
    # Create canvas
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid)
    
    # Try to use a better font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Place images in grid
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        
        x = col * img_width
        y = row * (img_height + label_height)
        
        # Paste image
        grid.paste(img, (x, y + label_height))
        
        # Draw label
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (img_width - text_width) // 2
        text_y = y + 5
        
        # Draw text with background
        draw.rectangle([text_x - 5, text_y - 2, text_x + text_width + 5, text_y + 30], fill='black')
        draw.text((text_x, text_y), label, fill='white', font=font)
    
    # Save grid
    grid.save(output_path, quality=95)
    print(f"\n✅ Comparison grid saved to: {output_path}")
    print(f"   Grid size: {grid_width}x{grid_height}")
    print(f"   Images: {len(images)}")
    
    # Also create a zoomed comparison of a region
    create_zoomed_comparison(images, labels, "comparison_zoomed.png")
    
    return output_path


def create_zoomed_comparison(images, labels, output_path, zoom_factor=2):
    """
    Create a zoomed-in comparison of the center region
    """
    print(f"\nCreating zoomed comparison ({zoom_factor}x)...")
    
    # Get center crop
    img_width, img_height = images[0].size
    crop_size = min(img_width, img_height) // 4  # 1/4 of the image
    
    left = (img_width - crop_size) // 2
    top = (img_height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    cropped_images = []
    for img in images:
        crop = img.crop((left, top, right, bottom))
        # Zoom in
        zoomed = crop.resize((crop_size * zoom_factor, crop_size * zoom_factor), Image.NEAREST)
        cropped_images.append(zoomed)
    
    # Create grid
    cols = min(3, len(cropped_images))
    rows = (len(cropped_images) + cols - 1) // cols
    
    label_height = 40
    zoomed_width = crop_size * zoom_factor
    zoomed_height = crop_size * zoom_factor
    
    grid_width = zoomed_width * cols
    grid_height = (zoomed_height + label_height) * rows
    
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, label) in enumerate(zip(cropped_images, labels)):
        row = idx // cols
        col = idx % cols
        
        x = col * zoomed_width
        y = row * (zoomed_height + label_height)
        
        grid.paste(img, (x, y + label_height))
        
        # Draw label
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (zoomed_width - text_width) // 2
        text_y = y + 5
        
        draw.rectangle([text_x - 5, text_y - 2, text_x + text_width + 5, text_y + 28], fill='black')
        draw.text((text_x, text_y), label, fill='white', font=font)
    
    grid.save(output_path, quality=95)
    print(f"✅ Zoomed comparison saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create side-by-side visual comparison')
    parser.add_argument("--input", required=True, help="Original input image")
    parser.add_argument("--comparison_dir", default="comparison_results", help="Directory with comparison outputs")
    parser.add_argument("--output", default="comparison_grid.png", help="Output grid image")
    
    args = parser.parse_args()
    
    create_side_by_side_comparison(args.input, args.comparison_dir, args.output)
