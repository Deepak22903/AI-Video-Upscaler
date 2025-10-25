import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
from srvgg_arch_improved import SRVGGNetImproved
import time


class ImagePairDataset(Dataset):
    """Dataset for loading HR images and creating LR pairs"""
    def __init__(self, hr_dir, patch_size=128, scale=2):
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale = scale
        self.lr_size = patch_size // scale
        
        # Get all image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            self.image_files.extend(list(self.hr_dir.rglob(ext)))
        
        print(f"Found {len(self.image_files)} images in {hr_dir}")
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files) * 10  # Multiple crops per image
    
    def __getitem__(self, idx):
        img_idx = idx % len(self.image_files)
        img_path = self.image_files[img_idx]
        
        # Load HR image
        hr_img = Image.open(img_path).convert('RGB')
        
        # Random crop
        w, h = hr_img.size
        if w < self.patch_size or h < self.patch_size:
            hr_img = hr_img.resize((max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC)
            w, h = hr_img.size
        
        left = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        top = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        hr_img = hr_img.crop((left, top, left + self.patch_size, top + self.patch_size))
        
        # Random augmentation
        if torch.rand(1) > 0.5:
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        if torch.rand(1) > 0.5:
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Create LR image
        lr_img = hr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        
        # Convert to tensor
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)
        
        return lr_tensor, hr_tensor


def train_improved_model(
    data_path,
    output_dir,
    num_epochs=30,
    batch_size=8,
    lr=1e-4,
    patch_size=128,
    num_feat=64,
    num_conv=12,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train improved ESRGAN model with better parameters
    
    Improvements:
    - Larger patches (64 -> 128) for better context
    - Larger batch size (4 -> 8) for better gradients  
    - More epochs (10 -> 30) for better convergence
    - Lower learning rate (2e-4 -> 1e-4) for stability
    - More features (32 -> 64) and layers (4 -> 12)
    - Mixed L1 + perceptual loss for better quality
    """
    
    print("="*60)
    print("IMPROVED MODEL TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Features: {num_feat}")
    print(f"  Conv layers: {num_conv}")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = SRVGGNetImproved(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=2,
        act_type='prelu'
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Estimated model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Create dataset and dataloader
    dataset = ImagePairDataset(data_path, patch_size=patch_size, scale=2)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Loss function - L1 loss for pixel-wise reconstruction
    criterion = nn.L1Loss()
    
    # Optimizer - Adam with lower learning rate for stability
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        epoch_start = time.time()
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / batch_count
                print(f"  Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {avg_loss:.6f}")
        
        # Epoch statistics
        avg_epoch_loss = epoch_loss / batch_count
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Average Loss: {avg_epoch_loss:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}\n")
        
        # Update learning rate scheduler
        scheduler.step(avg_epoch_loss)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f'model_improved_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}\n")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(output_dir, 'model_improved_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, best_path)
            print(f"✓ Best model saved: {best_path} (Loss: {best_loss:.6f})\n")
    
    print("="*60)
    print("Training completed!")
    print(f"Best loss achieved: {best_loss:.6f}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train improved ESRGAN model')
    parser.add_argument('--data_path', type=str, default='../../Real-ESRGAN/datasets/DIV2K/DIV2K_train_HR',
                        help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Training patch size')
    parser.add_argument('--num_feat', type=int, default=64,
                        help='Number of feature channels')
    parser.add_argument('--num_conv', type=int, default=12,
                        help='Number of convolutional layers')
    
    args = parser.parse_args()
    
    train_improved_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        num_feat=args.num_feat,
        num_conv=args.num_conv
    )
