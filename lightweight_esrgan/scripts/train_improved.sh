#!/bin/bash

# Quick Training Script for Improved Model
# This script trains with better parameters for higher quality

echo "=========================================="
echo "Training Improved ESRGAN Model"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Features: 64 (vs 32 lightweight)"
echo "  Layers: 12 (vs 4 lightweight)"
echo "  Patch size: 128 (vs 64 lightweight)"
echo "  Batch size: 8 (vs 4 lightweight)"
echo "  Epochs: 30 (vs 10 lightweight)"
echo "  Estimated model size: ~1.8 MB"
echo ""
echo "=========================================="
echo ""

cd /home/deepak/data/btechProject

python lightweight_esrgan/scripts/train_improved.py \
    --data_path Real-ESRGAN/datasets/DIV2K/DIV2K_train_HR \
    --output_dir lightweight_esrgan/models \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-4 \
    --patch_size 128 \
    --num_feat 64 \
    --num_conv 12

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
