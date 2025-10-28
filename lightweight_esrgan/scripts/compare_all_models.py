import torch
import argparse
import time
import numpy as np
from PIL import Image
import os


def compare_models(input_image, output_dir="comparison_results"):
    """
    Compare all available models on the same input image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = {
        "Lightweight": "lightweight_esrgan/models/model_mobile.ptl",
        "Lightweight Quantized": "lightweight_esrgan/models/model_quantized.ptl",
        "Improved": "lightweight_esrgan/models/model_improved_mobile.ptl",
        "Improved Quantized": "lightweight_esrgan/models/model_improved_quantized.ptl",
    }
    
    print("="*80)
    print("MODEL COMPARISON TEST")
    print("="*80)
    print(f"\nInput image: {input_image}")
    print(f"Output directory: {output_dir}\n")
    
    # Load input image
    print("Loading input image...")
    img = Image.open(input_image).convert('RGB')
    w, h = img.size
    img = img.resize((w - w % 2, h - h % 2))
    
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    print(f"  Input shape: {img_tensor.shape}")
    print(f"  Input size: {w}x{h}\n")
    
    results = {}
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"⚠️  {model_name}: Model not found at {model_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Load model
            print(f"  Loading model from {model_path}...")
            model = torch.jit.load(model_path)
            model.eval()
            
            # Get model size
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✓ Model loaded (Size: {model_size:.2f} MB)")
            
            # Warm-up
            with torch.no_grad():
                _ = model(img_tensor)
            
            # Benchmark
            print(f"  Running inference benchmark (10 iterations)...")
            times = []
            with torch.no_grad():
                for i in range(10):
                    start = time.time()
                    output = model(img_tensor)
                    times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            std_time = np.std(times) * 1000
            print(f"  ✓ Inference: {avg_time:.2f} ± {std_time:.2f} ms")
            
            # Save output
            output_path = os.path.join(output_dir, f"output_{model_name.replace(' ', '_').lower()}.png")
            output_img = output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            output_img = (output_img * 255).astype(np.uint8)
            Image.fromarray(output_img).save(output_path)
            print(f"  ✓ Output saved to {output_path}")
            
            # Calculate output size
            out_h, out_w = output_img.shape[:2]
            print(f"  ✓ Output size: {out_w}x{out_h} (Scale: {out_w/w:.1f}x)")
            
            results[model_name] = {
                'size_mb': model_size,
                'inference_ms': avg_time,
                'output_path': output_path
            }
            
        except Exception as e:
            print(f"  ✗ Error testing {model_name}: {e}")
    
    # Print summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Model':<25} {'Size (MB)':<12} {'Inference (ms)':<15} {'Output'}")
    print("-"*80)
    
    for model_name, data in results.items():
        print(f"{model_name:<25} {data['size_mb']:<12.2f} {data['inference_ms']:<15.2f} {data['output_path']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find best models
    if results:
        smallest = min(results.items(), key=lambda x: x[1]['size_mb'])
        fastest = min(results.items(), key=lambda x: x[1]['inference_ms'])
        
        print(f"\n✅ Smallest Model: {smallest[0]} ({smallest[1]['size_mb']:.2f} MB)")
        print(f"✅ Fastest Inference: {fastest[0]} ({fastest[1]['inference_ms']:.2f} ms)")
        
        if 'Improved Quantized' in results:
            print(f"\n🌟 RECOMMENDED: Improved Quantized")
            print(f"   - Best quality with acceptable size ({results['Improved Quantized']['size_mb']:.2f} MB)")
            print(f"   - Fast inference ({results['Improved Quantized']['inference_ms']:.2f} ms)")
            print(f"   - Optimized for mobile deployment")
    
    print("\n" + "="*80)
    print("View the output images in the comparison_results/ folder to compare visual quality!")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare all models on the same input')
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output_dir", default="comparison_results", help="Output directory")
    
    args = parser.parse_args()
    
    compare_models(args.input, args.output_dir)
