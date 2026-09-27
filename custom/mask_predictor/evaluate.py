import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import MaskDataset
from .io_utils import load_checkpoint
from .losses import combined_mask_loss, segmentation_metrics
from .model import TinyMaskPredictor
from .transforms import ImageMaskTransform


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained mask predictor.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, action='append', required=True)
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--split', choices=['train', 'val', 'all'], default='val')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--image-size', type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = MaskDataset(args.dataset_root,
                          split=args.split,
                          tasks=args.task or None,
                          val_fraction=args.val_fraction,
                          seed=args.seed,
                          transform=ImageMaskTransform(train=False,
                                                       image_size=args.image_size),
                          max_samples=args.max_samples)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = TinyMaskPredictor().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    totals = {
        'loss': 0.0,
        'bce': 0.0,
        'dice_loss': 0.0,
        'iou': 0.0,
        'dice': 0.0,
        'pixel_accuracy': 0.0,
    }
    batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            logits = model(images)
            loss, loss_parts = combined_mask_loss(logits, masks)
            metrics = segmentation_metrics(logits, masks)
            totals['loss'] += loss.item()
            totals['bce'] += loss_parts['bce']
            totals['dice_loss'] += loss_parts['dice_loss']
            totals['iou'] += metrics['iou']
            totals['dice'] += metrics['dice']
            totals['pixel_accuracy'] += metrics['pixel_accuracy']
            batches += 1
    result = {
        'num_samples': len(dataset),
        'num_batches': batches,
        'metrics': {key: value / max(1, batches) for key, value in totals.items()},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
