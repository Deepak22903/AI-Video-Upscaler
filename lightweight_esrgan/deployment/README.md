Mobile deployment notes

- `mobile_test.py` contains small helpers to run the scripted PyTorch model and a TFLite model locally for validation.

PyTorch Mobile flow
1. Train: `python scripts/train.py --lr_dir data/train/LR --hr_dir data/train/HR --save_dir models --epochs 50`
2. Quantize & script: `python scripts/quantize_torch_mobile.py --weights models/model_epoch_50.pth --out models/quantized_scripted.pt`
3. Push the `models/quantized_scripted.pt` to the mobile device and run with PyTorch Mobile APIs.

TFLite flow (optional)
1. Export ONNX: `python scripts/export_onnx.py --weights models/model_epoch_50.pth --out models/model.onnx`
2. Convert to TFLite: `python scripts/convert_onnx_to_tflite.py --onnx models/model.onnx --out models/model.tflite`
3. Run `deployment/mobile_test.py --tflite models/model.tflite --image test.png` to validate locally.

Notes
- ONNX->TFLite conversion requires `onnx-tf` and a compatible TensorFlow version.
- PyTorch dynamic quantization works best for linear layers; Conv2d quantization support may vary by PyTorch version.
