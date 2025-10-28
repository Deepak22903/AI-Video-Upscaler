import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import time
import numpy as np
from PIL import Image
from srvgg_arch_improved import SRVGGNetImproved
import math


class TiledInference:
    """
    Tiled inference for processing large images without memory issues
    Similar to Real-ESRGAN's tiling strategy
    """
    
    def __init__(self, model_path, tile_size=256, tile_pad=10, scale=2, device='cpu'):
        """
        Args:
            model_path: Path to the model file (.ptl or .pt)
            tile_size: Size of each tile (default: 256)
            tile_pad: Padding for each tile to avoid border artifacts (default: 10)
            scale: Upscaling factor (default: 2)
            device: Device to run inference on (default: 'cpu')
        """
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.scale = scale
        self.device = device
        
        print(f"Loading model from {model_path}...")
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        print("✓ Model loaded")
        
        # Get model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model size: {model_size:.2f} MB")
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Tile padding: {tile_pad}")
        print(f"  Scale factor: {scale}x")
    
    def process_tile(self, tile):
        """Process a single tile"""
        with torch.no_grad():
            output = self.model(tile)
        return output
    
    def enhance(self, img_tensor):
        """
        Enhanced inference with tiling for large images
        
        Args:
            img_tensor: Input image tensor (1, C, H, W)
        
        Returns:
            output_tensor: Enhanced image tensor (1, C, H*scale, W*scale)
        """
        batch, channel, height, width = img_tensor.shape
        
        # Calculate output size
        output_height = height * self.scale
        output_width = width * self.scale
        
        # Create output tensor
        output = torch.zeros(
            (batch, channel, output_height, output_width),
            dtype=img_tensor.dtype,
            device=self.device
        )
        
        # Calculate number of tiles
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        total_tiles = tiles_x * tiles_y
        
        print(f"\nProcessing image: {width}x{height}")
        print(f"Tiles: {tiles_x}x{tiles_y} = {total_tiles} tiles")
        
        tile_count = 0
        
        # Process each tile
        for i in range(tiles_y):
            for j in range(tiles_x):
                tile_count += 1
                print(f"\tTile {tile_count}/{total_tiles}", flush=True)
                
                # Calculate tile boundaries with padding
                ofs_x = j * self.tile_size
                ofs_y = i * self.tile_size
                
                # Input tile extraction with padding
                input_start_x = max(ofs_x - self.tile_pad, 0)
                input_end_x = min(ofs_x + self.tile_size + self.tile_pad, width)
                input_start_y = max(ofs_y - self.tile_pad, 0)
                input_end_y = min(ofs_y + self.tile_size + self.tile_pad, height)
                
                # Extract tile
                input_tile = img_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                
                # Process tile
                output_tile = self.process_tile(input_tile)
                
                # Calculate output positions
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale
                
                # Calculate the region to paste (removing padding effects)
                output_start_x_tile = (ofs_x - input_start_x) * self.scale
                output_end_x_tile = output_start_x_tile + min(self.tile_size, width - ofs_x) * self.scale
                output_start_y_tile = (ofs_y - input_start_y) * self.scale
                output_end_y_tile = output_start_y_tile + min(self.tile_size, height - ofs_y) * self.scale
                
                # Paste tile into output
                output[:, :, 
                       ofs_y * self.scale:(ofs_y + min(self.tile_size, height - ofs_y)) * self.scale,
                       ofs_x * self.scale:(ofs_x + min(self.tile_size, width - ofs_x)) * self.scale] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        
        return output


def process_image_tiled(model_path, input_path, output_path, tile_size=256, tile_pad=10, device='cpu'):
    """
    Process an image using tiled inference
    """
    print("="*60)
    print("TILED INFERENCE - IMPROVED MODEL")
    print("="*60)
    
    # Initialize tiled inference
    tiled_inferencer = TiledInference(
        model_path=model_path,
        tile_size=tile_size,
        tile_pad=tile_pad,
        scale=2,
        device=device
    )
    
    # Load input image
    print(f"\nLoading input image: {input_path}")
    img = Image.open(input_path).convert('RGB')
    w, h = img.size
    
    # Make dimensions divisible by tile size for cleaner processing
    # (optional, but can help with edge cases)
    img = img.resize((w - w % 2, h - h % 2))
    w, h = img.size
    
    print(f"  Input size: {w}x{h}")
    
    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    # Process with timing
    print("\nStarting inference...")
    start_time = time.time()
    
    output_tensor = tiled_inferencer.enhance(img_tensor)
    
    inference_time = (time.time() - start_time) * 1000
    
    # Convert back to image
    output_img = output_tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    output_img = (output_img * 255).astype(np.uint8)
    output_pil = Image.fromarray(output_img)
    
    # Save output
    print(f"\nSaving output to {output_path}...")
    output_pil.save(output_path)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Input size: {w}x{h}")
    print(f"Output size: {output_pil.size[0]}x{output_pil.size[1]}")
    print(f"Scale: {output_pil.size[0]/w:.1f}x")
    print(f"Inference time: {inference_time:.2f} ms ({inference_time/1000:.2f} seconds)")
    print(f"Output saved: {output_path}")
    print("="*60)
    
    return output_pil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiled inference for improved model')
    parser.add_argument("--model", required=True, help="Path to model (.ptl or .pt)")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--tile", type=int, default=256, help="Tile size (default: 256)")
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding (default: 10)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    
    process_image_tiled(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        tile_size=args.tile,
        tile_pad=args.tile_pad,
        device=args.device
    )
