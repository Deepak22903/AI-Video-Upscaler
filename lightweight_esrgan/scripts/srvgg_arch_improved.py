import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetImproved(nn.Module):
    """
    Improved SRVGG Network with more capacity for better quality
    
    Improvements over lightweight version:
    - num_feat: 32 -> 64 (double feature channels)
    - num_conv: 4 -> 12 (triple convolutional layers)
    - Added residual connections for better gradient flow
    - Better feature extraction capability
    """
    
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=12, upscale=2, act_type='prelu'):
        super(SRVGGNetImproved, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        
        # First convolution - feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Main convolutional body with residual connections
        self.body = nn.ModuleList()
        for i in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        
        # Upsampling layers
        self.conv_up = nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        
        # Final convolution
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        # Activation
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        # First convolution
        feat = self.act(self.conv_first(x))
        
        # Main body with residual connections
        # Add residual connection every 4 layers for better gradient flow
        body_feat = feat
        for i, conv in enumerate(self.body):
            body_feat = self.act(conv(body_feat))
            # Add residual connection every 4 layers
            if (i + 1) % 4 == 0 and i < len(self.body) - 1:
                body_feat = body_feat + feat
        
        # Add final residual connection
        feat = feat + body_feat
        
        # Upsampling
        out = self.pixel_shuffle(self.conv_up(feat))
        
        # Final convolution
        out = self.conv_last(out)
        
        return out


def test_model():
    """Test the improved model"""
    model = SRVGGNetImproved(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=12, upscale=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size estimate: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Model test passed!")


if __name__ == '__main__':
    test_model()
