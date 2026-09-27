import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def combined_mask_loss(logits, targets, bce_weight=0.5, dice_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    loss = (bce_weight * bce) + (dice_weight * dice)
    return loss, {'bce': bce.detach().item(), 'dice_loss': dice.detach().item()}


def segmentation_metrics(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    pred_sum = preds.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    pixel_accuracy = (preds == targets).float().mean(dim=(1, 2, 3))
    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'pixel_accuracy': pixel_accuracy.mean().item(),
    }
