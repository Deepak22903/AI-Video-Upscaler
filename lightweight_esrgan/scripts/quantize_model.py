import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from srvgg_arch_lightweight import SRVGGNetLightweight
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
import numpy as np


def quantize_model(weights_path, output_path, quantization_type='dynamic', device='cpu'):
    """
    Quantize PyTorch model for even better mobile performance
    
    Quantization types:
    - dynamic: Dynamic quantization (fastest conversion, good speedup)
    - static: Static quantization (requires calibration data, best performance)
    """
    print(f"Loading PyTorch model from {weights_path}...")
    model = SRVGGNetLightweight(num_in_ch=3, num_out_ch=3, num_feat=32, num_conv=4, upscale=2).to(device)
    
    ckpt = torch.load(weights_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print("✓ Model loaded")
    
    # Get original model size
    temp_path = "temp_original.pt"
    example_input = torch.randn(1, 3, 64, 64, device=device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(temp_path)
    original_size = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    
    print(f"\nOriginal model size: {original_size:.2f} MB")
    
    # Benchmark original model
    print("\nBenchmarking original model...")
    times_original = []
    with torch.no_grad():
        for _ in range(10):
            start = time.time()
            _ = model(example_input)
            times_original.append(time.time() - start)
    avg_time_original = np.mean(times_original) * 1000
    print(f"  Average inference time: {avg_time_original:.2f} ms")
    
    if quantization_type == 'dynamic':
        print("\n🔄 Applying dynamic quantization...")
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Conv2d, torch.nn.Linear},  # Quantize Conv2d and Linear layers
            dtype=torch.qint8
        )
        print("✓ Dynamic quantization applied")
        
        # Trace the quantized model
        print("\nTracing quantized model...")
        traced_quantized = torch.jit.trace(quantized_model, example_input)
        
        # Optimize for mobile
        print("Optimizing for mobile...")
        optimized_quantized = optimize_for_mobile(traced_quantized)
        
    elif quantization_type == 'static':
        print("\n🔄 Applying static quantization...")
        print("⚠️  Note: Static quantization requires calibration data")
        
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with example data (you should use real data here)
        print("Calibrating with example data...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(torch.randn(1, 3, 64, 64))
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        quantized_model = model
        print("✓ Static quantization applied")
        
        # Trace the quantized model
        print("\nTracing quantized model...")
        traced_quantized = torch.jit.trace(quantized_model, example_input)
        
        # Optimize for mobile
        print("Optimizing for mobile...")
        optimized_quantized = optimize_for_mobile(traced_quantized)
    
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    # Save quantized model
    print(f"\nSaving quantized model to {output_path}...")
    optimized_quantized._save_for_lite_interpreter(output_path)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Quantized model saved")
    
    # Also save regular quantized TorchScript
    torchscript_path = output_path.replace('.ptl', '_quantized.pt')
    traced_quantized.save(torchscript_path)
    ts_size = os.path.getsize(torchscript_path) / (1024 * 1024)
    
    # Benchmark quantized model
    print("\nBenchmarking quantized model...")
    times_quantized = []
    with torch.no_grad():
        for _ in range(10):
            start = time.time()
            _ = quantized_model(example_input)
            times_quantized.append(time.time() - start)
    avg_time_quantized = np.mean(times_quantized) * 1000
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 QUANTIZATION RESULTS")
    print("="*60)
    print(f"\n{'Metric':<30} {'Original':<15} {'Quantized':<15} {'Improvement'}")
    print("-"*60)
    print(f"{'Model Size (MB)':<30} {original_size:<15.2f} {quantized_size:<15.2f} {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"{'Inference Time (ms)':<30} {avg_time_original:<15.2f} {avg_time_quantized:<15.2f} {(1 - avg_time_quantized/avg_time_original)*100:.1f}%")
    print("-"*60)
    
    print(f"\n✅ Quantization complete!")
    print(f"\nOutput files:")
    print(f"  - Mobile: {output_path} ({quantized_size:.2f} MB)")
    print(f"  - Desktop: {torchscript_path} ({ts_size:.2f} MB)")
    
    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'original_time_ms': avg_time_original,
        'quantized_time_ms': avg_time_quantized,
        'size_reduction_percent': (1 - quantized_size/original_size) * 100,
        'speed_improvement_percent': (1 - avg_time_quantized/avg_time_original) * 100
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize PyTorch model for mobile deployment')
    parser.add_argument("--weights", required=True, help="Path to PyTorch weights (.pth)")
    parser.add_argument("--output", required=True, help="Output path for quantized mobile model (.ptl)")
    parser.add_argument("--type", default="dynamic", choices=['dynamic', 'static'], 
                        help="Quantization type: dynamic (default) or static")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    quantize_model(args.weights, args.output, args.type, args.device)
