"""
损失函数模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice损失实现（用于分割任务）"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """计算Dice损失"""
        # 展平张量
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # 计算Dice系数
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # Dice损失
        loss = 1 - dice

        return loss
