import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import MaskDataset
from .io_utils import ensure_dir, load_checkpoint
from .model import TinyMaskPredictor
from .transforms import ImageMaskTransform, resize_image_and_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with the tiny mask predictor.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-image', type=Path)
    parser.add_argument('--dataset-root', type=Path, action='append')
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image-size', type=int, default=180)
    return parser.parse_args()


def load_image_from_args(args):
    if args.input_image is not None:
        image = Image.open(args.input_image).convert('RGB')
        stem = args.input_image.stem
        return image, stem
    if args.dataset_root is None:
        raise ValueError('Either --input-image or --dataset-root must be provided')
    dataset = MaskDataset(args.dataset_root,
                          split='all',
                          tasks=args.task or None,
                          transform=ImageMaskTransform(train=False,
                                                       image_size=args.image_size),
                          image_size=args.image_size)
    sample = dataset[args.index]
    image = Image.open(sample['image_path']).convert('RGB')
    stem = Path(sample['image_path']).stem
    return image, stem


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = TinyMaskPredictor().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    image, stem = load_image_from_args(args)
    image, _ = resize_image_and_mask(image,
                                     Image.fromarray(np.zeros(image.size[::-1],
                                                              dtype=np.uint8)),
                                     image_size=args.image_size)
    image_tensor, _ = ImageMaskTransform(train=False,
                                         image_size=args.image_size)(image,
                                                                     Image.fromarray(
                                                                         np.zeros(
                                                                             image.size
                                                                             [::-1],
                                                                             dtype=np.uint8)))
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probs >= args.threshold).astype(np.uint8) * 255
    output_dir = ensure_dir(args.output_dir)
    Image.fromarray(mask).save(output_dir / f'{stem}_pred_mask.png')
    overlay = np.asarray(image, dtype=np.float32)
    overlay[mask > 0] = (0.65 * overlay[mask > 0]) + (
        0.35 * np.array([255.0, 0.0, 0.0], dtype=np.float32))
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        output_dir / f'{stem}_overlay.png')
    print(f'Saved prediction outputs into {output_dir}')


if __name__ == '__main__':
    main()
