import sys
sys.path.append('/home/deepak/data/btechProject/lightweight_esrgan')

from models.lightweight_model import LightweightESRGAN


import os
import argparse
from glob import glob
from PIL import Image
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from srvgg_arch_lightweight import SRVGGNetLightweight


class PairedImageDataset(Dataset):
    """Expect directory structure:
    data/train/LR/*.png
    data/train/HR/*.png
    filenames must match (or use same ordering).
    """
    def __init__(self, lr_dir, hr_dir, crop_size=64):
        self.lr_files = sorted(glob(os.path.join(lr_dir, '*')))
        self.hr_files = sorted(glob(os.path.join(hr_dir, '*')))
        assert len(self.lr_files) == len(self.hr_files), "LR and HR counts must match"
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_files[idx]).convert('RGB')
        hr = Image.open(self.hr_files[idx]).convert('RGB')
        # random crop
        w, h = lr.size
        cw = min(self.crop_size, w)
        ch = min(self.crop_size, h)
        if w == cw or h == ch:
            x = 0
            y = 0
        else:
            x = random.randint(0, w - cw)
            y = random.randint(0, h - ch)
        lr_patch = lr.crop((x, y, x + cw, y + ch))
        # HR patch should be appropriately sized (scale 2 assumed)
        hr_patch = hr.crop((x*2, y*2, (x+cw)*2, (y+ch)*2))
        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)


import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = SRVGGNetLightweight(num_in_ch=3, num_out_ch=3, num_feat=32, num_conv=4, upscale=args.scale).to(device)
    logger.info("Model initialized.")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = PairedImageDataset(args.lr_dir, args.hr_dir, crop_size=args.crop_size)
    # Limit the dataset to 100 samples
    limited_dataset = torch.utils.data.Subset(dataset, range(100))
    loader = DataLoader(limited_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    logger.info(f"Dataset loaded with {len(limited_dataset)} samples.")

    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Saving models to: {args.save_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        logger.info(f"Starting epoch {epoch}/{args.epochs}.")

        for i, (lr, hr) in enumerate(loader, 1):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                logger.info(f"Epoch {epoch} [{i}/{len(loader)}] - Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_dir', default='data/train/LR')
    parser.add_argument('--hr_dir', default='data/train/HR')
    parser.add_argument('--save_dir', default='models')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=20)
    args = parser.parse_args()
    train(args)
