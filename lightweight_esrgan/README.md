# Lightweight Real-ESRGAN Training Pipeline

This project is designed to train a lightweight Real-ESRGAN model optimized for mobile deployment. The pipeline includes data preprocessing, model training, quantization, and deployment.

## Directory Structure
- `data/`: Contains training and validation datasets.
- `models/`: Stores model checkpoints and configurations.
- `scripts/`: Includes training and utility scripts.
- `deployment/`: Contains files for deploying the model to mobile devices.

## Steps
1. Prepare the dataset and place it in the `data/` directory.
2. Modify the model architecture in the `models/` directory.
3. Use the scripts in `scripts/` to train and evaluate the model.
4. Quantize the trained model to INT8.
5. Deploy the model using the files in `deployment/`.

## Requirements
- Python 3.11
- PyTorch
- TensorFlow Lite
- ONNX

Refer to the `requirements.txt` file for the complete list of dependencies.