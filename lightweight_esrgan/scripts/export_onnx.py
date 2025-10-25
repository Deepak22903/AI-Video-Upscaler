import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import os
from models.lightweight_model import LightweightESRGAN
from srvgg_arch_lightweight import SRVGGNetLightweight


def export_onnx(weights, out, input_size=(1,3,64,64), device='cpu'):
    model = SRVGGNetLightweight(num_in_ch=3, num_out_ch=3, num_feat=32, num_conv=4, upscale=2).to(device)
    ckpt = torch.load(weights, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    torch.onnx.export(model, dummy, out, opset_version=18, input_names=['input'], output_names=['output'])
    print(f"Exported ONNX model to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', default='models/model.onnx')
    parser.add_argument('--input_h', type=int, default=64)
    parser.add_argument('--input_w', type=int, default=64)
    args = parser.parse_args()
    export_onnx(args.weights, args.out, input_size=(1,3,args.input_h,args.input_w))
