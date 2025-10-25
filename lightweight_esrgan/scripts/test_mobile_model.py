import torch
import argparse
import time
import numpy as np
from PIL import Image
import os


def test_mobile_model(model_path, input_image=None, output_path=None):
    """
    Test PyTorch Mobile model for inference
    """
    print(f"Loading mobile model from {model_path}...")
    
    # Load the mobile model
    model = torch.jit.load(model_path)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Get model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model size: {model_size:.2f} MB")
    
    # Create test input
    if input_image:
        print(f"\nLoading input image: {input_image}")
        img = Image.open(input_image).convert('RGB')
        # Resize to ensure dimensions are divisible by upscale factor
        w, h = img.size
        img = img.resize((w - w % 2, h - h % 2))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        print(f"  Input shape: {img_tensor.shape}")
    else:
        print("\nUsing random test input (64x64)...")
        img_tensor = torch.randn(1, 3, 64, 64)
    
    # Warm-up
    print("\nWarming up...")
    with torch.no_grad():
        _ = model(img_tensor)
    
    # Benchmark inference
    print("\nRunning inference benchmark (10 iterations)...")
    times = []
    with torch.no_grad():
        for i in range(10):
            start = time.time()
            output = model(img_tensor)
            end = time.time()
            times.append(end - start)
            print(f"  Iteration {i+1}: {(end-start)*1000:.2f} ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n✓ Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Output shape: {output.shape}")
    
    # Save output if requested
    if output_path and input_image:
        print(f"\nSaving output to {output_path}...")
        output_img = output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        output_img = (output_img * 255).astype(np.uint8)
        Image.fromarray(output_img).save(output_path)
        print("✓ Output saved")
    
    # Memory usage estimation
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Model size: {model_size:.2f} MB")
    print(f"  Inference time: {avg_time*1000:.2f} ms")
    
    print("\n✅ Mobile model test completed successfully!")
    
    return {
        'model_size_mb': model_size,
        'avg_inference_ms': avg_time * 1000,
        'param_count': param_count,
        'output_shape': output.shape
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test PyTorch Mobile model')
    parser.add_argument("--model", required=True, help="Path to mobile model (.ptl or .pt)")
    parser.add_argument("--input", help="Optional: input image to test")
    parser.add_argument("--output", help="Optional: output path to save result")
    
    args = parser.parse_args()
    
    test_mobile_model(args.model, args.input, args.output)
