import torch
import argparse
import os
from models.lightweight_model import LightweightESRGAN


def quantize_and_script(weights, out_path='models/quantized_scripted.pt', device='cpu'):
    model = LightweightESRGAN()
    ckpt = torch.load(weights, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Dynamic quantization (weights) for linear and conv layers where supported
    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)
    # Script and save for PyTorch Mobile
    scripted = torch.jit.script(qmodel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted.save(out_path)
    print(f"Saved quantized scripted model to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', default='models/quantized_scripted.pt')
    args = parser.parse_args()
    quantize_and_script(args.weights, args.out)
