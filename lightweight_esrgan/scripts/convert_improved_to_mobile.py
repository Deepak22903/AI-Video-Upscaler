import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from srvgg_arch_improved import SRVGGNetImproved
from torch.utils.mobile_optimizer import optimize_for_mobile


def convert_improved_to_pytorch_mobile(weights_path, output_path, device='cpu'):
    """
    Convert improved PyTorch model to PyTorch Mobile format (.ptl)
    """
    print("Loading improved PyTorch model...")
    model = SRVGGNetImproved(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=12, upscale=2).to(device)
    
    ckpt = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print("✓ Model loaded")
    
    print("\nConverting to TorchScript...")
    # Create example input
    example_input = torch.randn(1, 3, 128, 128, device=device)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    print("✓ Model traced")
    
    print("\nOptimizing for mobile...")
    # Optimize for mobile
    optimized_model = optimize_for_mobile(traced_model)
    print("✓ Model optimized")
    
    print(f"\nSaving to {output_path}...")
    # Save the model
    optimized_model._save_for_lite_interpreter(output_path)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"✓ PyTorch Mobile model saved")
    print(f"  Size: {file_size:.2f} MB")
    
    # Also save regular TorchScript version
    torchscript_path = output_path.replace('.ptl', '.pt')
    traced_model.save(torchscript_path)
    ts_size = os.path.getsize(torchscript_path) / 1024 / 1024
    print(f"\n✓ TorchScript model also saved to {torchscript_path}")
    print(f"  Size: {ts_size:.2f} MB")
    
    print("\n✅ Conversion complete!")
    print("\nUsage:")
    print(f"  - For mobile: {output_path}")
    print(f"  - For desktop/server: {torchscript_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert improved PyTorch model to PyTorch Mobile format')
    parser.add_argument("--weights", required=True, help="Path to PyTorch weights (.pth)")
    parser.add_argument("--output", required=True, help="Output path for mobile model (.ptl)")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    convert_improved_to_pytorch_mobile(args.weights, args.output, args.device)
