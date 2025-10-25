"""
Small utilities to test the exported models locally (PyTorch Mobile and TFLite)
"""
import os
import numpy as np
from PIL import Image


def run_torch_mobile(model_path, image_path):
    import torch
    from torchvision import transforms
    model = torch.jit.load(model_path)
    model.eval()
    img = Image.open(image_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    x = to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    y = y.squeeze(0).clamp(0,1).mul(255).permute(1,2,0).cpu().numpy().astype('uint8')
    out = Image.fromarray(y)
    out.save('out_torch_mobile.png')
    print('Saved out_torch_mobile.png')


def run_tflite(model_path, image_path):
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError('TensorFlow is required to run TFLite models') from e
    from tensorflow.lite.python.interpreter import Interpreter
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    # assume model expects NHWC float32
    arr = np.expand_dims(arr, 0)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]['index'])
    y = np.clip(y[0], 0, 1) * 255.0
    y = y.astype('uint8')
    out = Image.fromarray(y)
    out.save('out_tflite.png')
    print('Saved out_tflite.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', help='path to scripted torch model')
    parser.add_argument('--tflite', help='path to tflite model')
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    if args.torch:
        run_torch_mobile(args.torch, args.image)
    if args.tflite:
        run_tflite(args.tflite, args.image)
