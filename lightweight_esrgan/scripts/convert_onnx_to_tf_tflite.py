import argparse
import tensorflow as tf
import onnx
from onnx2tf import convert
import os
import shutil

def convert_onnx_to_tf_tflite(onnx_path, tf_saved_model_dir, tflite_path):
    print("Loading ONNX model...")
    # Convert ONNX to TensorFlow using onnx2tf
    print("\nConverting ONNX to TensorFlow...")
    try:
        # Clean up existing directory
        if os.path.exists(tf_saved_model_dir):
            shutil.rmtree(tf_saved_model_dir)
        
        # Convert ONNX model to TensorFlow SavedModel using onnx2tf
        convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=os.path.dirname(tf_saved_model_dir),
            output_signaturedefs=True,
            output_h5=False,
            output_weight_and_json=False,
            output_integer_quantized_tflite=False,
            output_float16_quantized_tflite=False,
            output_dynamic_range_quantized_tflite=False,
        )
        print("✓ Conversion successful")
    except Exception as e:
        print(f"✗ ONNX to TensorFlow conversion failed: {e}")
        print("\nTrying alternative method with basic onnx2tf...")
        try:
            # Simple conversion without extra options
            convert(input_onnx_file_path=onnx_path)
            print("✓ Conversion successful with basic settings")
        except Exception as e2:
            print(f"✗ Alternative conversion also failed: {e2}")
            return

    print("\nConverting TensorFlow SavedModel to TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)

        # Enable TensorFlow ops for complex models
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        # Allow custom ops
        converter.allow_custom_ops = True

        # Try with optimizations first
        print("  Attempting with optimizations...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        file_size = len(tflite_model) / 1024 / 1024
        print(f"✓ TFLite model saved to {tflite_path}")
        print(f"  Size: {file_size:.2f} MB")

    except Exception as e:
        print(f"  Failed with optimizations: {e}")
        print("  Retrying without optimizations...")

        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.allow_custom_ops = True

            tflite_model = converter.convert()

            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            file_size = len(tflite_model) / 1024 / 1024
            print(f"✓ TFLite model saved to {tflite_path}")
            print(f"  Size: {file_size:.2f} MB")

        except Exception as e2:
            print(f"✗ TFLite conversion failed: {e2}")
            print("\n⚠️  Note: ESRGAN models may have operations not fully supported in TFLite.")
            print("   The TensorFlow SavedModel was created successfully and can be used for inference.")
            print("   For mobile deployment, consider using ONNX Runtime Mobile instead.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorFlow and TFLite')
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--tf_saved_model_dir", required=True, help="Output directory for TF SavedModel")
    parser.add_argument("--tflite", required=True, help="Output path for TFLite model")

    args = parser.parse_args()

    convert_onnx_to_tf_tflite(args.onnx, args.tf_saved_model_dir, args.tflite)