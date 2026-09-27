import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import MaskDataset
from .io_utils import append_jsonl, ensure_dir, save_checkpoint, save_json
from .losses import combined_mask_loss, segmentation_metrics
from .model import TinyMaskPredictor
from .transforms import ImageMaskTransform


def _serialize_config(args):
    def serialize(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [serialize(item) for item in value]
        return value

    return {key: serialize(value) for key, value in vars(args).items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a small binary mask predictor.')
    parser.add_argument('--dataset-root', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-val-samples', type=int)
    parser.add_argument('--horizontal-flip-prob', type=float, default=0.0)
    parser.add_argument('--image-size', type=int, default=180)
    return parser.parse_args()


def make_dataloaders(args):
    tasks = args.task or None
    train_dataset = MaskDataset(
        args.dataset_root,
        split='train',
        tasks=tasks,
        val_fraction=args.val_fraction,
        seed=args.seed,
        transform=ImageMaskTransform(train=True,
                                     horizontal_flip_prob=args.horizontal_flip_prob,
                                     image_size=args.image_size),
        max_samples=args.max_train_samples)
    val_dataset = MaskDataset(args.dataset_root,
                              split='val',
                              tasks=tasks,
                              val_fraction=args.val_fraction,
                              seed=args.seed,
                              transform=ImageMaskTransform(train=False,
                                                           image_size=args.image_size),
                              max_samples=args.max_val_samples)
    if len(train_dataset) == 0:
        raise ValueError('Training dataset is empty')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers) if len(val_dataset) > 0 else None
    return train_dataset, val_dataset, train_loader, val_loader


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    totals = {
        'loss': 0.0,
        'bce': 0.0,
        'dice_loss': 0.0,
        'iou': 0.0,
        'dice': 0.0,
        'pixel_accuracy': 0.0,
    }
    num_batches = 0
    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        logits = model(images)
        loss, loss_parts = combined_mask_loss(logits, masks)
        metrics = segmentation_metrics(logits.detach(), masks)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        totals['loss'] += loss.detach().item()
        totals['bce'] += loss_parts['bce']
        totals['dice_loss'] += loss_parts['dice_loss']
        totals['iou'] += metrics['iou']
        totals['dice'] += metrics['dice']
        totals['pixel_accuracy'] += metrics['pixel_accuracy']
        num_batches += 1
    return {key: value / max(1, num_batches) for key, value in totals.items()}


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / 'checkpoints')
    config = _serialize_config(args)
    save_json(output_dir / 'train_config.json', config)

    train_dataset, val_dataset, train_loader, val_loader = make_dataloaders(args)
    device = torch.device(args.device)
    model = TinyMaskPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    best_val_iou = float('-inf')
    summary = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'best_val_iou': None,
        'best_epoch': None,
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        record = {'epoch': epoch, 'train': train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, optimizer=None, device=device)
            record['val'] = val_metrics
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                summary['best_val_iou'] = best_val_iou
                summary['best_epoch'] = epoch
                save_checkpoint(checkpoints_dir / 'best.pt',
                                model,
                                optimizer,
                                epoch,
                                config,
                                record)
        append_jsonl(output_dir / 'metrics.jsonl', record)
        save_checkpoint(checkpoints_dir / 'last.pt',
                        model,
                        optimizer,
                        epoch,
                        config,
                        record)
    save_json(output_dir / 'summary.json', summary)
    print(
        f"Finished training tiny mask predictor with {len(train_dataset)} train samples and {len(val_dataset)} val samples into {output_dir}"
    )


if __name__ == '__main__':
    main()
