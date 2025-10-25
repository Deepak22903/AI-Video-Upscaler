"""
Convert ONNX -> TensorFlow SavedModel -> TFLite
Requires: onnx, onnx-tf, tensorflow

Usage:
  python convert_onnx_to_tflite.py --onnx models/model.onnx --out models/model.tflite
"""
import argparse
import os


def convert(onnx_path, out_path):
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("Missing dependencies for ONNX->TFLite conversion. Install onnx, onnx-tf, and tensorflow.") from e

    model = onnx.load(onnx_path)
    tf_rep = prepare(model)  # creates a TF representation
    saved = out_path + '.saved_model'
    tf_rep.export_graph(saved)

    # Convert saved model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Converted ONNX -> TFLite and wrote {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    convert(args.onnx, args.out)
