import argparse
import json
from pathlib import Path

from PIL import Image

from .io_utils import prepare_task_output


def parse_args():
    parser = argparse.ArgumentParser(
        description='Resize an exported RGB/mask dataset to a square target size.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--image-size', type=int, default=180)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resize_task(task_dir, output_dir, image_size, overwrite=False):
    manifest_path = task_dir / 'manifest.jsonl'
    if not manifest_path.exists():
        return 0
    layout = prepare_task_output(output_dir, task_dir.name, include_previews=False)
    if layout.manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f'Manifest already exists at {layout.manifest_path}; pass --overwrite to replace it.'
        )
    count = 0
    with manifest_path.open(encoding='utf-8') as src, layout.manifest_path.open(
            'w', encoding='utf-8') as dst:
        for line in src:
            record = json.loads(line)
            image = Image.open(task_dir / record['image_path']).convert('RGB')
            mask = Image.open(task_dir / record['mask_path']).convert('L')
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
            image.save(layout.task_dir / record['image_path'])
            mask.save(layout.task_dir / record['mask_path'])
            dst.write(json.dumps(record, sort_keys=True) + '\n')
            count += 1
    return count


def main():
    args = parse_args()
    if args.output_dir.exists() and args.overwrite:
        for manifest in args.output_dir.glob('*/manifest.jsonl'):
            manifest.unlink()
    total = 0
    for task_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        total += resize_task(task_dir,
                             args.output_dir,
                             image_size=args.image_size,
                             overwrite=args.overwrite)
    print(
        f'Resized {total} RGB/mask pairs from {args.input_dir} into {args.output_dir} at {args.image_size}x{args.image_size}'
    )


if __name__ == '__main__':
    main()
