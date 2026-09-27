import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from custom.mask_predictor.dataset import MaskDataset, load_manifest_records, split_records
from custom.mask_predictor.io_utils import load_checkpoint, save_checkpoint
from custom.mask_predictor.losses import combined_mask_loss, segmentation_metrics
from custom.mask_predictor.model import TinyMaskPredictor
from custom.mask_predictor.predict_dreamer4 import (concat_frames, dilate_mask,
                                                    output_strip_dir,
                                                    process_strip,
                                                    save_frame_sequences,
                                                    split_episode_strip)
from custom.mask_predictor.transforms import ImageMaskTransform


def _write_sample(task_dir, episode_index, frame_index, foreground_box):
    images_dir = task_dir / 'images'
    masks_dir = task_dir / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    stem = f'episode_{episode_index:05d}_frame_{frame_index:06d}'
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[..., 1] = 64
    x0, y0, x1, y1 = foreground_box
    image[y0:y1, x0:x1, 0] = 255
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(image).save(images_dir / f'{stem}.png')
    Image.fromarray(mask).save(masks_dir / f'{stem}.png')
    return {
        'task_name': task_dir.name,
        'episode_index': episode_index,
        'frame_index': frame_index,
        'image_path': f'images/{stem}.png',
        'mask_path': f'masks/{stem}.png',
    }


class MaskPredictorTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.tmpdir.name) / 'dataset'
        task_dirs = [self.dataset_root / 'walker_walk', self.dataset_root / 'hopper_hop']
        for task_index, task_dir in enumerate(task_dirs):
            records = []
            for frame_index in range(5):
                records.append(
                    _write_sample(task_dir,
                                  episode_index=0,
                                  frame_index=frame_index,
                                  foreground_box=(20 + frame_index, 30, 60, 80)))
            manifest_path = task_dir / 'manifest.jsonl'
            with manifest_path.open('w', encoding='utf-8') as handle:
                for record in records:
                    handle.write(json.dumps(record) + '\n')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_manifest_records(self):
        records = load_manifest_records(self.dataset_root)
        self.assertEqual(len(records), 10)
        self.assertTrue(records[0]['image_abspath'].endswith('.png'))

    def test_load_manifest_records_from_multiple_roots(self):
        second_root = Path(self.tmpdir.name) / 'dataset_b'
        task_dir = second_root / 'cheetah_jump'
        record = _write_sample(task_dir,
                               episode_index=1,
                               frame_index=0,
                               foreground_box=(10, 10, 40, 40))
        with (task_dir / 'manifest.jsonl').open('w', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        records = load_manifest_records([self.dataset_root, second_root])
        self.assertEqual(len(records), 11)
        self.assertIn('cheetah_jump', {entry['task_name'] for entry in records})

    def test_split_records_per_task(self):
        records = load_manifest_records(self.dataset_root)
        train_records = split_records(records, split='train', val_fraction=0.2, seed=1)
        val_records = split_records(records, split='val', val_fraction=0.2, seed=1)
        self.assertEqual(len(train_records), 8)
        self.assertEqual(len(val_records), 2)
        self.assertEqual(sorted({record['task_name'] for record in val_records}),
                         ['hopper_hop', 'walker_walk'])

    def test_dataset_returns_expected_tensors(self):
        dataset = MaskDataset(self.dataset_root, split='all')
        sample = dataset[0]
        self.assertEqual(tuple(sample['image'].shape), (3, 180, 180))
        self.assertEqual(tuple(sample['mask'].shape), (1, 180, 180))
        self.assertTrue(sample['image'].dtype.is_floating_point)
        self.assertGreater(sample['mask'].sum().item(), 0.0)

    def test_dataset_accepts_multiple_roots(self):
        second_root = Path(self.tmpdir.name) / 'dataset_c'
        task_dir = second_root / 'fish_swim'
        record = _write_sample(task_dir,
                               episode_index=0,
                               frame_index=0,
                               foreground_box=(5, 5, 20, 20))
        with (task_dir / 'manifest.jsonl').open('w', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        dataset = MaskDataset([self.dataset_root, second_root], split='all')
        self.assertEqual(len(dataset), 11)

    def test_model_output_shape(self):
        model = TinyMaskPredictor()
        logits = model(torch.randn(2, 3, 180, 180))
        self.assertEqual(tuple(logits.shape), (2, 1, 180, 180))

    def test_transform_resizes_to_target_shape(self):
        image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        mask = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
        image_tensor, mask_tensor = ImageMaskTransform(train=False,
                                                       image_size=180)(image, mask)
        self.assertEqual(tuple(image_tensor.shape), (3, 180, 180))
        self.assertEqual(tuple(mask_tensor.shape), (1, 180, 180))

    def test_combined_loss_and_metrics(self):
        logits = torch.zeros(2, 1, 16, 16)
        targets = torch.zeros(2, 1, 16, 16)
        targets[:, :, 2:6, 3:8] = 1.0
        loss, parts = combined_mask_loss(logits, targets)
        metrics = segmentation_metrics(logits, targets)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn('bce', parts)
        self.assertIn('iou', metrics)

    def test_checkpoint_roundtrip(self):
        model = TinyMaskPredictor()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_path = Path(self.tmpdir.name) / 'checkpoint.pt'
        save_checkpoint(checkpoint_path,
                        model,
                        optimizer,
                        epoch=3,
                        config={'lr': 1e-3},
                        metrics={'iou': 0.5})
        checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
        self.assertEqual(checkpoint['epoch'], 3)
        self.assertEqual(checkpoint['config']['lr'], 1e-3)

    def test_split_and_concat_episode_strip(self):
        frame_a = Image.fromarray(np.full((8, 8, 3), 10, dtype=np.uint8))
        frame_b = Image.fromarray(np.full((8, 8, 3), 20, dtype=np.uint8))
        strip = concat_frames([frame_a, frame_b])
        frames, frame_size = split_episode_strip(strip)
        self.assertEqual(frame_size, 8)
        self.assertEqual(len(frames), 2)
        self.assertEqual(np.asarray(frames[0])[0, 0, 0], 10)
        self.assertEqual(np.asarray(frames[1])[0, 0, 0], 20)

    def test_dilate_mask_adds_context_pixels(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        dilated = dilate_mask(mask, context_pixels=1)
        self.assertEqual(int(dilated.sum()), 9)
        self.assertTrue(dilated[3, 3])
        self.assertTrue(dilated[5, 5])

    def test_save_frame_sequences_writes_indexed_pngs(self):
        strip_dir = Path(self.tmpdir.name) / 'strip'
        frames = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(3)]
        masks = [Image.fromarray(np.zeros((8, 8), dtype=np.uint8)) for _ in range(3)]
        saved = save_frame_sequences(strip_dir,
                                     frames,
                                     masks,
                                     frames,
                                     frames,
                                     save_every=2)
        self.assertEqual(saved, 2)
        self.assertTrue((strip_dir / 'frames' / '000000.png').exists())
        self.assertTrue((strip_dir / 'frames' / '000002.png').exists())
        self.assertFalse((strip_dir / 'frames' / '000001.png').exists())

    def test_process_strip_saves_resized_frame_sequences(self):
        strip_path = Path(self.tmpdir.name) / 'episode.png'
        frame_a = Image.fromarray(np.full((8, 8, 3), 10, dtype=np.uint8))
        frame_b = Image.fromarray(np.full((8, 8, 3), 20, dtype=np.uint8))
        concat_frames([frame_a, frame_b]).save(strip_path)
        args = SimpleNamespace(output_dir=Path(self.tmpdir.name) / 'outputs',
                               input_dir=None,
                               threshold=0.5,
                               image_size=180,
                               output_frame_size='4',
                               context_pixels=0,
                               save_frame_sequences=True,
                               save_every=1,
                               batch_size=2)
        fake_masks = [
            np.ones((180, 180), dtype=bool),
            np.ones((180, 180), dtype=bool),
        ]
        with mock.patch('custom.mask_predictor.predict_dreamer4.predict_masks',
                        return_value=fake_masks):
            process_strip(strip_path, model=None, args=args, device='cpu')
        with Image.open(args.output_dir / 'episode' / 'frames' / '000000.png') as saved_frame:
            self.assertEqual(saved_frame.size, (4, 4))
        with Image.open(args.output_dir / 'episode' / 'masks' / '000000.png') as saved_mask:
            self.assertEqual(saved_mask.size, (4, 4))

    def test_output_strip_dir_preserves_relative_input_path(self):
        output_dir = Path('/tmp/masked')
        input_dir = Path('/data/raw-dreamer4')
        path = input_dir / 'mixed-large' / 'walker-walk-0.png'
        strip_dir = output_strip_dir(output_dir, path, input_dir)
        self.assertEqual(strip_dir,
                         output_dir / 'mixed-large' / 'walker-walk-0')


if __name__ == '__main__':
    unittest.main()
