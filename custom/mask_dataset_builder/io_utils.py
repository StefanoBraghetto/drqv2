import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TaskOutputLayout:
    task_dir: Path
    images_dir: Path
    masks_dir: Path
    previews_dir: Path | None
    manifest_path: Path


def prepare_task_output(output_root, task_name, include_previews=False):
    task_dir = Path(output_root) / task_name
    images_dir = task_dir / 'images'
    masks_dir = task_dir / 'masks'
    previews_dir = task_dir / 'previews' if include_previews else None
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if previews_dir is not None:
        previews_dir.mkdir(parents=True, exist_ok=True)
    return TaskOutputLayout(task_dir=task_dir,
                            images_dir=images_dir,
                            masks_dir=masks_dir,
                            previews_dir=previews_dir,
                            manifest_path=task_dir / 'manifest.jsonl')


def frame_stem(episode_index, frame_index):
    return f'episode_{episode_index:05d}_frame_{frame_index:06d}'


def write_rgb(path, rgb):
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def write_mask(path, mask):
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


class ManifestWriter:
    def __init__(self, manifest_path, overwrite=False):
        self.manifest_path = Path(manifest_path)
        self.overwrite = overwrite
        self._handle = None

    def __enter__(self):
        if self.manifest_path.exists() and not self.overwrite:
            raise FileExistsError(
                f'Manifest already exists at {self.manifest_path}. '
                'Pass overwrite=True to replace it.')
        mode = 'w' if self.overwrite else 'x'
        self._handle = self.manifest_path.open(mode, encoding='utf-8')
        return self

    def write(self, record):
        json.dump(record, self._handle, sort_keys=True)
        self._handle.write('\n')
        self._handle.flush()

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.close()
        return False
