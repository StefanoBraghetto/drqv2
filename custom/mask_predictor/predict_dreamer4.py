import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .io_utils import ensure_dir, load_checkpoint, save_json
from .model import TinyMaskPredictor
from .transforms import ImageMaskTransform, resize_image_and_mask

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the mask predictor over Dreamer-style episode strips.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-png', type=Path)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--pattern', default='*.png')
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--image-size', type=int, default=180)
    parser.add_argument('--output-frame-size', default='source')
    parser.add_argument('--context-pixels', type=int, default=0)
    parser.add_argument('--save-frame-sequences', action='store_true')
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def parse_output_frame_size(value, source_frame_size):
    if value == 'source':
        return source_frame_size
    size = int(value)
    if size <= 0:
        raise ValueError(f'output frame size must be positive, got {value!r}')
    return size


def split_episode_strip(image):
    width, height = image.size
    if width % height != 0:
        raise ValueError(
            f'Expected a horizontal strip with width multiple of height, got {image.size}'
        )
    frame_size = height
    num_frames = width // frame_size
    frames = []
    for frame_index in range(num_frames):
        left = frame_index * frame_size
        frames.append(image.crop((left, 0, left + frame_size, frame_size)))
    return frames, frame_size


def concat_frames(frames):
    if not frames:
        raise ValueError('Need at least one frame to concatenate')
    width, height = frames[0].size
    strip = Image.new(frames[0].mode, (width * len(frames), height))
    for index, frame in enumerate(frames):
        strip.paste(frame, (index * width, 0))
    return strip


def dilate_mask(mask, context_pixels):
    if context_pixels <= 0:
        return mask
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = (context_pixels * 2) + 1
    dilated = F.max_pool2d(tensor,
                           kernel_size=kernel,
                           stride=1,
                           padding=context_pixels)
    return (dilated[0, 0].numpy() > 0.5)


def overlay_mask(image, mask):
    rgb = np.asarray(image, dtype=np.float32)
    overlay = rgb.copy()
    overlay[mask] = (0.65 * overlay[mask]) + (
        0.35 * np.array([255.0, 0.0, 0.0], dtype=np.float32))
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def apply_mask(image, mask):
    rgb = np.asarray(image, dtype=np.uint8).copy()
    rgb[~mask] = 0
    return Image.fromarray(rgb)


def save_frame_sequences(strip_dir,
                         output_frames,
                         mask_frames,
                         masked_frames,
                         overlay_frames,
                         save_every=1):
    if save_every <= 0:
        raise ValueError(f'save_every must be positive, got {save_every}')
    frames_dir = ensure_dir(strip_dir / 'frames')
    masks_dir = ensure_dir(strip_dir / 'masks')
    masked_dir = ensure_dir(strip_dir / 'masked_frames')
    overlay_dir = ensure_dir(strip_dir / 'overlay_frames')
    saved = 0
    for frame_index, (frame, mask, masked, overlay) in enumerate(
            zip(output_frames, mask_frames, masked_frames, overlay_frames)):
        if frame_index % save_every != 0:
            continue
        stem = f'{frame_index:06d}.png'
        frame.save(frames_dir / stem)
        mask.save(masks_dir / stem)
        masked.save(masked_dir / stem)
        overlay.save(overlay_dir / stem)
        saved += 1
    return saved


def predict_masks(model, frames, image_size, threshold, batch_size, device):
    tensors = []
    zero_mask = Image.fromarray(np.zeros((image_size, image_size), dtype=np.uint8))
    transform = ImageMaskTransform(train=False, image_size=image_size)
    for frame in frames:
        resized_frame, _ = resize_image_and_mask(frame, zero_mask, image_size=image_size)
        frame_tensor, _ = transform(resized_frame, zero_mask)
        tensors.append(frame_tensor)
    masks = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start:start + batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
        masks.extend(prob >= threshold for prob in probs)
    return masks


def maybe_filter_path(path, tasks):
    if not tasks:
        return True
    path_text = str(path).replace('-', '_').lower()
    return any(task.lower() in path_text for task in tasks)


def iter_input_paths(args):
    if args.input_png is not None:
        return [args.input_png]
    if args.input_dir is None:
        raise ValueError('Either --input-png or --input-dir must be provided')
    return sorted(
        path for path in args.input_dir.rglob(args.pattern)
        if path.is_file() and maybe_filter_path(path, args.task))


def output_strip_dir(output_dir, path, input_dir=None):
    if input_dir is None:
        return output_dir / path.stem
    relative_path = path.relative_to(input_dir)
    return output_dir / relative_path.parent / path.stem


def process_strip(path, model, args, device):
    source_image = Image.open(path).convert('RGB')
    source_frames, source_frame_size = split_episode_strip(source_image)
    masks = predict_masks(model,
                          source_frames,
                          image_size=args.image_size,
                          threshold=args.threshold,
                          batch_size=args.batch_size,
                          device=device)
    output_frame_size = parse_output_frame_size(args.output_frame_size,
                                                source_frame_size)
    output_frames = []
    mask_frames = []
    masked_frames = []
    overlay_frames = []
    for frame, mask in zip(source_frames, masks):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        frame_image = frame
        if output_frame_size != args.image_size:
            mask_image = mask_image.resize((output_frame_size, output_frame_size),
                                           Image.Resampling.NEAREST)
        if frame.size[0] != output_frame_size:
            frame_image = frame.resize((output_frame_size, output_frame_size),
                                       Image.Resampling.BILINEAR)
        output_frames.append(frame_image)
        mask_bool = (np.asarray(mask_image, dtype=np.uint8) > 127)
        mask_bool = dilate_mask(mask_bool, args.context_pixels)
        mask_image = Image.fromarray(mask_bool.astype(np.uint8) * 255)
        mask_frames.append(mask_image)
        masked_frames.append(apply_mask(frame_image, mask_bool))
        overlay_frames.append(overlay_mask(frame_image, mask_bool))

    strip_dir = ensure_dir(output_strip_dir(args.output_dir, path, args.input_dir))
    concat_frames(mask_frames).save(strip_dir / 'mask_strip.png')
    concat_frames(masked_frames).save(strip_dir / 'masked_strip.png')
    concat_frames(overlay_frames).save(strip_dir / 'overlay_strip.png')
    saved_frame_count = 0
    if args.save_frame_sequences:
        saved_frame_count = save_frame_sequences(strip_dir,
                                                output_frames,
                                                mask_frames,
                                                masked_frames,
                                                overlay_frames,
                                                save_every=args.save_every)
    save_json(strip_dir / 'metadata.json', {
        'source_path': str(path),
        'num_frames': len(source_frames),
        'source_frame_size': source_frame_size,
        'model_image_size': args.image_size,
        'output_frame_size': output_frame_size,
        'context_pixels': args.context_pixels,
        'threshold': args.threshold,
        'save_frame_sequences': args.save_frame_sequences,
        'save_every': args.save_every,
        'saved_frame_count': saved_frame_count,
    })
    return len(source_frames), strip_dir


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = TinyMaskPredictor().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    paths = iter_input_paths(args)
    if not paths:
        raise ValueError('No input PNGs matched the requested inputs')
    total_frames = 0
    for path in paths:
        frame_count, strip_dir = process_strip(path, model, args, device)
        total_frames += frame_count
        print(f'Processed {path} -> {strip_dir} ({frame_count} frames)')
    print(f'Processed {len(paths)} strip(s) and {total_frames} total frame(s)')


if __name__ == '__main__':
    main()
