import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from custom.mask_dataset_builder.io_utils import ManifestWriter, frame_stem, prepare_task_output, write_mask, write_rgb
from custom.mask_dataset_builder.export_dataset import _time_step_scalar
from custom.mask_dataset_builder.preview import overlay_mask
from custom.mask_dataset_builder.random_rollout import iter_random_rollout, sample_random_action
from custom.mask_dataset_builder.render_pairs import resolve_geom_ids, segmentation_to_mask
from custom.mask_dataset_builder.task_specs import TASK_MASK_SPECS, configured_task_names


class _FakeTimeStep:
    def __init__(self, last=False, reward=0.0, discount=1.0):
        self._last = last
        self.reward = reward
        self.discount = discount

    def last(self):
        return self._last


class _FakeEnv:
    def __init__(self):
        self._step_count = 0
        self._action_spec = SimpleNamespace(shape=(2, ),
                                            minimum=np.array([-1.0, -0.5]),
                                            maximum=np.array([1.0, 0.5]),
                                            dtype=np.float32)

    def reset(self):
        self._step_count = 0
        return _FakeTimeStep(last=False)

    def action_spec(self):
        return self._action_spec

    def step(self, action):
        del action
        self._step_count += 1
        return _FakeTimeStep(last=self._step_count >= 2, reward=1.0)


class _FakeModel:
    ngeom = 4
    nbody = 3

    def __init__(self):
        self.geom_bodyid = np.array([0, 1, 1, 2], dtype=np.int32)
        self._names = {
            ('geom', 0): 'floor',
            ('geom', 1): 'walker_torso',
            ('geom', 2): 'walker_foot',
            ('geom', 3): 'target_ball',
            ('body', 0): 'world',
            ('body', 1): 'walker',
            ('body', 2): 'target',
        }

    def id2name(self, index, kind):
        return self._names.get((kind, index))


class MaskDatasetBuilderTests(unittest.TestCase):
    def test_every_configured_task_has_spec(self):
        configured = configured_task_names()
        missing = sorted(set(configured) - set(TASK_MASK_SPECS))
        self.assertEqual(missing, [])

    def test_sample_random_action_respects_bounds(self):
        spec = SimpleNamespace(shape=(3, ),
                               minimum=np.array([-1.0, -2.0, 0.0]),
                               maximum=np.array([1.0, 2.0, 3.0]),
                               dtype=np.float32)
        action = sample_random_action(spec, np.random.default_rng(3))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.all(action >= spec.minimum))
        self.assertTrue(np.all(action <= spec.maximum))

    def test_rollout_includes_reset_frame(self):
        frames = list(
            iter_random_rollout(_FakeEnv(),
                                num_episodes=1,
                                max_steps_per_episode=4,
                                seed=1))
        self.assertEqual([frame.frame_index for frame in frames], [0, 1, 2])
        self.assertIsNone(frames[0].action)
        self.assertIsNotNone(frames[1].action)

    def test_resolve_geom_ids_include_all_and_exclude_background(self):
        spec = TASK_MASK_SPECS['walker_walk']
        geom_ids = resolve_geom_ids(_FakeModel(), spec)
        self.assertEqual(geom_ids, (1, 2, 3))

    def test_resolve_geom_ids_body_and_geom_fragments(self):
        from custom.mask_dataset_builder.task_specs import TaskMaskSpec

        spec = TaskMaskSpec(task_name='demo',
                            description='demo',
                            include_geom_name_fragments=('ball', ),
                            include_body_name_fragments=('walker', ),
                            exclude_geom_name_fragments=('floor', ))
        geom_ids = resolve_geom_ids(_FakeModel(), spec)
        self.assertEqual(geom_ids, (1, 2, 3))

    def test_segmentation_to_mask_filters_geom_ids(self):
        segmentation = np.array([[[1, 5], [2, 5]], [[2, 1], [3, 5]]],
                                dtype=np.int32)
        mask = segmentation_to_mask(segmentation, (2, 3))
        np.testing.assert_array_equal(mask,
                                      np.array([[False, True], [False, True]]))

    def test_prepare_output_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = prepare_task_output(tmpdir, 'walker_walk', include_previews=True)
            image_path = layout.images_dir / f'{frame_stem(0, 0)}.png'
            mask_path = layout.masks_dir / f'{frame_stem(0, 0)}.png'
            write_rgb(image_path, np.zeros((4, 4, 3), dtype=np.uint8))
            write_mask(mask_path, np.array([[0, 1], [1, 0]], dtype=bool))
            with ManifestWriter(layout.manifest_path, overwrite=True) as manifest:
                manifest.write({'image_path': 'images/example.png'})
            self.assertTrue(image_path.exists())
            self.assertTrue(mask_path.exists())
            lines = layout.manifest_path.read_text().splitlines()
            self.assertEqual(json.loads(lines[0])['image_path'], 'images/example.png')
            self.assertTrue(layout.previews_dir.exists())

    def test_overlay_mask_tints_selected_pixels(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.array([[False, True], [False, False]])
        overlaid = overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.5)
        self.assertEqual(tuple(overlaid[0, 1]), (127, 0, 0))

    def test_time_step_scalar_handles_none(self):
        self.assertEqual(_time_step_scalar(None, 0.0), 0.0)
        self.assertEqual(_time_step_scalar(None, 1.0), 1.0)
        self.assertEqual(_time_step_scalar(2.5, 0.0), 2.5)


if __name__ == '__main__':
    unittest.main()
