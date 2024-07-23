import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .UNet import build_UNet


class MSCMR(nn.Module):
    def __init__(self, freeze_whst=False):
        super().__init__()

        if freeze_whst:
            for p in self.parameters():
                p.requires_grad_(False)
        self.tasks = 'MR'
        self.UNet = build_UNet()

    def forward(self, samples: torch.Tensor, task):
        seg_masks = self.UNet(samples)
        out = {"pred_masks": seg_masks}
        return out
