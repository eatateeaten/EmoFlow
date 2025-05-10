"""
LPIPS loss implementation for perceptual similarity measurement.
"""

import torch
import torch.nn as nn
import lpips

class LPIPSLoss(nn.Module):
    def __init__(self, device, net='vgg'):
        super(LPIPSLoss, self).__init__()
        # Initialize LPIPS with VGG backbone
        self.lpips = lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    def forward(self, generated, target):
        """
        Compute LPIPS loss between generated and target images.
        Args:
            generated: Generated image from CFM (batch_size, 3, H, W), in [0, 1]
            target: Target expressive image (batch_size, 3, H, W), in [0, 1]
        Returns:
            loss: Scalar LPIPS loss
        """
        # Convert grayscale to RGB if needed
        if generated.shape[1] == 1:
            generated = generated.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
            
        # LPIPS expects images in [-1, 1]
        generated = generated * 2 - 1
        target = target * 2 - 1
        return self.lpips(generated, target).mean()
