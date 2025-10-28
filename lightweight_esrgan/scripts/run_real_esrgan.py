import sys
import os
sys.path.append('/home/deepak/data/btechProject/Real-ESRGAN-fresh')

import torch
import argparse
import time
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def run_real_esrgan(input_image, output_path, model_path, scale=4, tile=0, tile_pad=10):
    """
    Run Real-ESRGAN on input image
    """
    print("="*60)
    print("Running Real-ESRGAN (Original)")
    print("="*60)
    
    # Load model
    print(f"\nLoading Real-ESRGAN model from {model_path}...")
    
    # RealESRGAN x4plus model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=False  # Use FP32 for CPU
    )
    
    print("✓ Model loaded")
    
    # Get model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model size: {model_size:.2f} MB")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Load input image
    print(f"\nLoading input image: {input_image}")
    img = Image.open(input_image).convert('RGB')
    img_np = np.array(img)
    
    print(f"  Input size: {img.size[0]}x{img.size[1]}")
    
    # Run inference with timing
    print("\nRunning inference...")
    start_time = time.time()
    
    output, _ = upsampler.enhance(img_np, outscale=4)
    
    inference_time = (time.time() - start_time) * 1000
    print(f"✓ Inference completed in {inference_time:.2f} ms")
    
    # Save output
    print(f"\nSaving output to {output_path}...")
    output_img = Image.fromarray(output)
    output_img.save(output_path)
    
    print(f"✓ Output saved")
    print(f"  Output size: {output_img.size[0]}x{output_img.size[1]}")
    print(f"  Scale: {output_img.size[0] / img.size[0]:.1f}x")
    
    print("\n" + "="*60)
    print("Real-ESRGAN Results")
    print("="*60)
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Parameters: {total_params:,}")
    print(f"Inference Time: {inference_time:.2f} ms")
    print(f"Output: {output_path}")
    print("="*60 + "\n")
    
    return {
        'model_size_mb': model_size,
        'parameters': total_params,
        'inference_ms': inference_time,
        'output_path': output_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Real-ESRGAN on input image')
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--model", default="/home/deepak/data/btechProject/Real-ESRGAN-fresh/weights/RealESRGAN_x4plus.pth",
                        help="Path to Real-ESRGAN model")
    parser.add_argument("--tile", type=int, default=0, help="Tile size (0 for no tiling)")
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding")
    
    args = parser.parse_args()
    
    run_real_esrgan(args.input, args.output, args.model, tile=args.tile, tile_pad=args.tile_pad)
